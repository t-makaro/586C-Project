#include "cunn.cuh"

CUNN::CUNN(std::vector<int> layers) : layers(layers) {
    numLayers = layers.size();
    weights.reserve(numLayers - 1);
    biases.reserve(numLayers - 1);
    activations.reserve(numLayers);

    /*d_weights.reserve(numLayers - 1);
    d_biases.reserve(numLayers - 1);
    d_activations.reserve(numLayers);

    */

    // initialize device pointer vecotrs
	d_weights.resize(numLayers - 1);
	d_biases.resize(numLayers - 1);
	d_activations.resize(numLayers);
}

CUNN::~CUNN() {}

// alloc device weights, biases, activations
void CUNN::deviceAlloc() {
    size_t sizeA0 = layers[0] * sizeof(float); // input vector
    cudaMalloc(&d_activations[0], sizeA0);

    for (int i = 1; i < numLayers; i++) {
        int M = layers[i-1];
        int N = layers[i];
        
        size_t sizeWi = M * N * sizeof(float);
        size_t sizeAi = N * sizeof(float);
        size_t sizeBi = N * sizeof(float);

		cudaMalloc(&d_weights[i - 1], sizeWi);
		cudaMalloc(&d_biases[i - 1], sizeBi);
		cudaMalloc(&d_activations[i], sizeAi);
    }
}

void CUNN::copyParametersToDevice() {
    deviceAlloc();

    for (int i = 0; i < weights.size(); i++) {
        int M = weights[i].size();
        int N = weights[i][0].size();
        assert(M == biases[i].size());

        size_t sizeWi = M * N * sizeof(float);
        std::vector<float> Wi_flattened(M * N);
        for (int j = 0; j < M; j++) {
            std::copy(weights[i][j].begin(), weights[i][j].end(), Wi_flattened.begin() + j * N);
        }

        cudaMemcpy(d_weights[i], Wi_flattened.data(), sizeWi, cudaMemcpyHostToDevice);

        size_t sizeBi = M * sizeof(float);
        cudaMemcpy(d_biases[i], biases[i].data(), sizeBi, cudaMemcpyHostToDevice);
    }
}

void CUNN::copyWeights(const std::vector<Matrix> weights) {
    assert(weights.size() == numLayers - 1);
    for (int i = 0; i < weights.size(); i++) {
        assert(weights[i].size() == layers[i + 1]);
        assert(weights[i][0].size() == layers[i]);
    }
    this->weights = weights;
}

void CUNN::copyBiases(const std::vector<Vector> biases) {
    assert(biases.size() == numLayers - 1);
    for (int i = 0; i < biases.size(); i++) {
        assert(biases[i].size() == layers[i + 1]);
    }
    this->biases = biases;
}

float CUNN::sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
float CUNN::d_sigmoid(float x) {
    float xp = exp(-x);
    return xp / ((1.0 + xp) * (1.0 + xp));
}

Vector& CUNN::forward(const Vector& x, Vector& result) {
    activations[0] = x;
    for (int i = 1; i < numLayers; i++) {
        forwardLayer(weights[i - 1], biases[i - 1], activations[i - 1],
            activations[i]);
    }
    result = activations[numLayers - 1];
    return result;
}

Vector& CUNN::forwardLayer(const Matrix& w, const Vector& b, const Vector& a,
    Vector& result) {
    // return sigmoid(add(multiply(w, a, result), b, result));
    return cu_utility::cuForwardLayer(w, b, a, result);
}

void CUNN::train(const float* d_trainingData, const int* d_trainingLabels, 
    const int M, const int N, const int iterations, const int batchSize,
    float learningRate) {
    for (int j = 0; j < iterations; j++) {
        for (int i = 0; i < M; i += batchSize) {
            updateFromBatch(d_trainingData+i*N, d_trainingLabels+i, batchSize, N, learningRate);
        }
    }
}

std::vector<float*> CUNN::allocate_like_weights() {
    std::vector<float*> d_Weights;

    // Allocate zeros to accumulate the gradiant over the batch.
    for (int i = 0; i < numLayers - 1; i++) {
        // Allocate memory on the GPU for change in weights
        float* temp_weights;

        size_t weightSize = layers[i + 1] * layers[i] * sizeof(float);

        // Allocate GPU memory
        cudaMalloc(&temp_weights, weightSize);

        // Initialize the allocated memory to 0.0 (optional, but often needed)
        cudaMemset(temp_weights, 0.0, weightSize);

        // Store pointers in vectors
        d_Weights.push_back(temp_weights);
    }
    return d_Weights;
}
std::vector<float*> CUNN::allocate_like_biases() {
    std::vector<float*> d_Biases;

    // Allocate zeros to accumulate the gradiant over the batch.
    for (int i = 0; i < numLayers - 1; i++) {
        // Allocate memory on the GPU for change in weights and biases
        float* temp_biases;

        size_t biasSize = layers[i + 1] * sizeof(float);

        // Allocate GPU memory
        cudaMalloc(&temp_biases, biasSize);

        // Initialize the allocated memory to 0.0 (optional, but often needed)
        cudaMemset(temp_biases, 0.0, biasSize);

        // Store pointers in vectors
        d_Biases.push_back(temp_biases);
    }
    return d_Biases;
}
void deallocateVector(std::vector<float*> vec) {
    for (int i = 0; i < vec.size(); i++) {
        cudaFree(vec[0]);
    }
}

void CUNN::updateFromBatch(const float* d_batch, const int* d_labels, 
    const int batchSize, const int N, const float learningRate) {

    std::vector<float*> d_ddWeights = allocate_like_weights();
    std::vector<float*> d_ddBiases = allocate_like_biases();

    // calculate individual gradiants and average them together
    for (int i = 0; i < batchSize; i++) {
        std::vector<float*> d_dWeights = allocate_like_weights();
        std::vector<float*> d_dBiases = allocate_like_biases();
        //backwards(d_dWeights, d_dBiases, batch+i*n, labels+i);
        for (int j = 0; j < numLayers-1; j++) {
            //add(d_ddWeights[j], d_dWeights[j], d_ddWeights[j], 1.0 / batchSize);
            //add(d_ddBiases[j], d_dBiases[j], d_ddBiases[j], 1.0 / batchSize);
        }
        deallocateVector(d_dWeights);
        deallocateVector(d_dBiases);
    }
    // update the weights and biases with gradient computed above at the learning rate
    for (int i = 0; i < numLayers-1; i++) {
        //add(d_weights[j], d_ddWeights[j], d_weights[j], -learningRate);
        //add(d_biases[j], d_ddBiases[j], d_biases[j], -learningRate);
    }

    deallocateVector(d_ddWeights);
    deallocateVector(d_ddBiases);
}

void CUNN::backwards(std::vector<Matrix>& dWeights_output,
    std::vector<Vector>& dBiases_output,
    const Vector& testData, int testLabel) {
    activations[0] = testData;
    for (int i = 1; i < numLayers; i++) {
        forwardZ(weights[i - 1], biases[i - 1], activations[i - 1], zs[i]);
        sigmoid(zs[i], activations[i]);
    }
    Vector delta;
    for (int i = 0; i < numLayers - 1; i++) {
        if (i == 0) {
            cost_derivative(activations[activations.size() - 1], testLabel,
                delta);
        }
        else {
            activation_derivative(weights[numLayers - i], zs[numLayers - i],
                delta);
        }
        multiply_elementwise(d_sigmoid(zs[numLayers - 1 - i]), delta,
            dBiases_output[numLayers - 1 - i]);
        outer_product(activations[numLayers - 2 - i],
            dBiases_output[numLayers - 1 - i],
            dWeights_output[numLayers - 1 - i]);
    }
}

void CUNN::cost_derivative(const Vector& last_activation, const int label,
    Vector& result) {
    for (int i = 0; i < 10; i++) {
        if (i == label) {
            result[i] = -1 / last_activation[i];
        }
        else {
            result[i] = 1 / (1 - last_activation[i]);
        }
    }
    return;
}

Matrix& CUNN::outer_product(const Vector& a, const Vector& b, Matrix& result) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < b.size(); j++) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}

void CUNN::activation_derivative(const Matrix& weights, Vector& z,
    Vector& previous) {
    d_sigmoid(z);
    multiply_elementwise(z, previous, previous);
    Matrix temp;
    transpose(weights, temp);
    Vector result;
    multiply(temp, previous, result);
    previous = result;
}

void CUNN::transpose(const Matrix& a, Matrix& result) {
    // Get the dimensions of the input matrix 'a'
    int rows = a.size();
    int cols = a[0].size();

    // Resize the result matrix to hold the transpose (cols x rows)
    result.resize(cols);
    for (int i = 0; i < cols; ++i) {
        result[i].resize(rows);
    }

    // Perform the transpose operation
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[j][i] = a[i][j];
        }
    }
}

float CUNN::evaluate(const Matrix& testData,
    const std::vector<int>& testLabels) {
    int numCorrect = 0;

    // timing
    auto start = std::chrono::high_resolution_clock::now();

    //Vector result(10, 0);
    //for (int i = 0; i < testData.size(); i++) {
    //    Vector input = testData[i];
    //    Vector output = forward(input, result);

    //    int maxIndex = 0;
    //    float maxVal = 0;
    //    for (int j = 0; j < output.size(); j++) {
    //        if (output[j] > maxVal) {
    //            maxVal = output[j];
    //            maxIndex = j;
    //        }
    //    }
    //    if (maxIndex == testLabels[i]) {
    //        numCorrect++;
    //    }
    //}

    // should be testLabels.size() x 10
    // reserve memory for predictions
	Matrix predictions(testLabels.size(), Vector(10, 0));
	cu_utility::cuForward(d_weights, d_biases, d_activations, layers, testData, predictions);

    for (int i = 0; i < predictions.size(); i++) {
        Vector pred = predictions[i];
        int maxIndex = 0;
        float maxVal = 0;
        for (int j = 0; j < pred.size(); j++) {
            if (pred[j] > maxVal) {
                maxVal = pred[j];
                maxIndex = j;
            }
        }
        if (maxIndex == testLabels[i]) {
            numCorrect++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "done." << std::endl;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds."
        << std::endl;
    float accuracy = (float)numCorrect / testData.size();
    std::cout << "Train Accuracy: " << accuracy << std::endl;
    return accuracy;
}

Vector& CUNN::forwardZ(const Matrix& w, const Vector& b, const Vector& a,
    Vector& result) {
    return add(multiply(w, a, result), b, result);
}

Vector& CUNN::multiply(const Matrix& w, const Vector& x, Vector& result) {
    cu_utility::cuMatMulVector(w, x, result);
    return result;
}

Vector& CUNN::multiply_elementwise(const Vector& a, const Vector& b,
    Vector& result) {
    for (int i = 0; i < a.size(); i++) {
        result[i] = a[i] * b[i];
    }
    return result;
}

Vector& CUNN::add(const Vector& x, const Vector& b, Vector& result) {
    return cu_utility::cuVectorAdd(x, b, result);
}

Vector& CUNN::add(const Vector& x, const Vector& b, Vector& result,
    const float scale) {
    assert(x.size() == b.size() && x.size() == result.size());

    for (int i = 0; i < x.size(); i++) {
        result[i] = x[i] + b[i] * scale;
    }
    return result;
}

Matrix& CUNN::add(const Matrix& x, const Matrix& b, Matrix& result,
    const float scale) {
    assert(x.size() == b.size() && x.size() == result.size());

    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < x[0].size(); j++) {
            result[i][j] = x[i][j] + b[i][j] * scale;
        }
    }
    return result;
}

Vector& CUNN::sigmoid(const Vector& x, Vector& result) {
    result.assign(x.begin(), x.end());
    return cu_utility::cuSigmoid(result);
}

Vector& CUNN::sigmoid(Vector& x) { return cu_utility::cuSigmoid(x); }

Vector& CUNN::d_sigmoid(Vector& x) { return cu_utility::cuDSigmoid(x); }

