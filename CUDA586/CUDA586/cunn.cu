#include "cunn.cuh"

CUNN::CUNN(std::vector<int> layers) : layers(layers) {
    numLayers = layers.size();
    weights.reserve(numLayers - 1);
    biases.reserve(numLayers - 1);
    activations.reserve(numLayers);
    d_zs.reserve(numLayers);

    /*d_weights.reserve(numLayers - 1);
    d_biases.reserve(numLayers - 1);
    d_activations.reserve(numLayers);

    */

    // initialize device pointer vecotrs
	d_weights.resize(numLayers - 1);
	d_biases.resize(numLayers - 1);
	d_activations.resize(numLayers);
    d_zs.resize(numLayers);

	d_activations_batch.resize(numLayers);
}

CUNN::~CUNN() {
	deviceFree();
}

// alloc device weights, biases, activations
void CUNN::deviceAlloc() {
    size_t sizeA0 = layers[0] * sizeof(float); // input vector
    cudaMalloc(&d_activations[0], sizeA0);
    cudaMalloc(&d_zs[0], sizeA0);

    for (int i = 1; i < numLayers; i++) {
        int M = layers[i-1];
        int N = layers[i];
        
        size_t sizeWi = M * N * sizeof(float);
        size_t sizeAi = N * sizeof(float);
        size_t sizeBi = N * sizeof(float);

		cudaMalloc(&d_weights[i - 1], sizeWi);
		cudaMalloc(&d_biases[i - 1], sizeBi);
		cudaMalloc(&d_activations[i], sizeAi);
        cudaMalloc(&d_zs[i], sizeAi);
    }
}

void CUNN::deviceFree() {
    cudaFree(d_activations[0]);
    cudaFree(d_zs[0]);
    for (int i = 1; i < numLayers; i++) {
        cudaFree(d_weights[i - 1]);
        cudaFree(d_biases[i - 1]);
        cudaFree(d_activations[i]);
        cudaFree(d_zs[i]);
    }
}

void CUNN::setBatchSizeDevice(int batchSize) {
	this->batchSize = batchSize;

    size_t sizeA0 = layers[0] * batchSize * sizeof(float); // input batch 
    cudaMalloc(&d_activations_batch[0], sizeA0);

    for (int i = 1; i < numLayers; i++) {
        int M = layers[i-1];
        int N = layers[i];
        
        size_t sizeAi = N * sizeof(float) * batchSize;
		cudaMalloc(&d_activations_batch[i], sizeAi);
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

void CUNN::testForwardZ(bool isGpu, Vector &testData)
{
    zs.reserve(numLayers);
    for (int i = 0; i < numLayers; i++) {
        activations.push_back(Vector(layers[i], 0.0));
        zs.push_back(Vector(layers[i], 0.0));
        if (i < numLayers - 1) {
            dWeights.push_back(Matrix(layers[i + 1], Vector(layers[i], 0.0)));
            dBiases.push_back(Vector(layers[i + 1], 0.0));
        }
    }

    int i = 1; // Run the first layer as a test
    if(isGpu)
    {
        // No alloc needed here, we called copyParametersToDevice outside
        float* d_test;
        size_t s = testData.size() * sizeof(float);
        cudaMalloc(&d_test, s);
        cudaMemcpy(d_test, testData.data(),s, cudaMemcpyHostToDevice);
        cudaMemcpy(d_activations[0], d_test, s, cudaMemcpyDeviceToDevice);

        int N = layers[i - 1];
        int M = layers[i];
        cu_utility::cuForwardLayerWithZs(d_weights[i - 1], d_biases[i - 1], d_activations[i - 1], d_zs[i], d_activations[i], M, N);
        cudaDeviceSynchronize();
        cudaMemcpy(zs[i].data(),d_zs[i], zs[i].size() * sizeof(float), cudaMemcpyDeviceToHost);
        deviceFree();
        cudaFree(d_test);
    }
    else
    {
        activations[0] = testData;
	    forwardZ(weights[i - 1], biases[i - 1], activations[i - 1], zs[i]);
        sigmoid(zs[i], activations[i]);
    }
    //cu_utility::printVector(zs[i], 10); // Breakpoint here to see
}

void CUNN::testBackwardOutputLayer(bool isGPU, Vector& testData, int testLabel)
{
    //testForwardZ(isGPU, testData);

    zs.reserve(numLayers);
    for (int i = 0; i < numLayers; i++) {
        activations.push_back(Vector(layers[i], 0.0));
        zs.push_back(Vector(layers[i], 0.0));
        if (i < numLayers - 1) {
            dWeights.push_back(Matrix(layers[i + 1], Vector(layers[i], 0.0)));
            dBiases.push_back(Vector(layers[i + 1], 0.0));
        }
    }
    activations[0] = testData;

    Vector dBiases_tOutput(10, 0);
    Matrix dWeights_tOutput(10, Vector(300, 0));
    Vector dWeights_tFlattened(3000, 0);

    Vector dBiases_tOutput2(300, 0);
    Matrix dWeights_tOutput2(300, Vector(300, 0));
    Vector dWeights_tFlattened2(90000, 0);
    if(isGPU)
    {
        std::cout << "GPU Output: \n";
        float* d_test;
        size_t s = testData.size() * sizeof(float);
        cudaMalloc(&d_test, s);
        cudaMemcpy(d_test, testData.data(), s, cudaMemcpyHostToDevice);
        cudaMemcpy(d_activations[0], d_test, s, cudaMemcpyDeviceToDevice);

        // Stage 1: Forward Z pass
        for (int i = 1; i < numLayers; i++)
        {
            int N = layers[i - 1];
            int M = layers[i];
            cu_utility::cuForwardLayerWithZs(d_weights[i - 1], d_biases[i - 1], d_activations[i - 1], d_zs[i], d_activations[i], M, N);
            //cudaDeviceSynchronize();
            cudaMemcpy(zs[i].data(), d_zs[i], zs[i].size() * sizeof(float), cudaMemcpyDeviceToHost);
        }
        // Stage 2: Cost Derivative Pass (output layer)
        cu_utility::printVector(zs[numLayers - 1], 10); // zs is correct
        std::vector<float*> d_delta = allocate_like_biases(); // delta.size = zsi.size for each layer i.e. like weight
        float* d_biasOutput, * d_weightOutput;
        int* d_testLabel;
        size_t f_size = sizeof(float);
        cudaMalloc(&d_biasOutput, f_size * 10);
        cudaMalloc(&d_weightOutput, f_size * 3000);
        cudaMalloc(&d_testLabel, sizeof(int));
        cudaMemcpy(d_testLabel, &testLabel,sizeof(int), cudaMemcpyHostToDevice);

        cu_utility::cuBackwardOutputLayer(d_activations[numLayers - 1], d_activations[numLayers - 2], d_biasOutput, d_weightOutput,
            d_zs[numLayers - 1], d_delta[numLayers - 2], d_testLabel, layers[2], layers[3]);


        cudaDeviceSynchronize();
        cudaMemcpy(dBiases_tOutput.data(), d_biasOutput, f_size * 10, cudaMemcpyDeviceToHost);
        cudaMemcpy(dWeights_tFlattened.data(), d_weightOutput, f_size * 3000, cudaMemcpyDeviceToHost);

        // Vector sliced_vec(dWeights_tFlattened.begin(), dWeights_tFlattened.begin() + 301);
        //cu_utility::printVector(sliced_vec, 10);
        
        

        // Stage 3: One Regular Pass
        float* d_biasOutput2, * d_weightOutput2;
        cudaMalloc(&d_biasOutput2, f_size * 300);
        cudaMalloc(&d_weightOutput2, f_size * 90000);
        cudaDeviceSynchronize();
        cu_utility::cuBackwardRegularLayer(d_activations[1], d_biasOutput2, d_weights[numLayers-2], d_weightOutput2, d_zs[numLayers - 2], d_zs[numLayers - 1],
            d_delta[numLayers - 3], d_delta[numLayers - 2], layers[1], layers[2], layers[3]);
        auto delta_in = Vector(300, 0);
        auto delta_out = Vector(10, 0);
        cudaMemcpy(delta_out.data(), d_delta[numLayers - 2], 10 * f_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(delta_in.data(), d_delta[numLayers - 3], 300 * f_size, cudaMemcpyDeviceToHost);
        //cu_utility::printVector(delta_out, 10); // delta_out is correct, delta_in is wrong, indicating transMul kernel issue
        //cu_utility::printVector(delta_in, 10);
        Vector weightVector(3000, 0);
        cudaMemcpy(weightVector.data(), d_weights[numLayers - 2], 3000 * f_size, cudaMemcpyDeviceToHost);

        //Vector sliced_vec(weightVector.begin() + 1200, weightVector.begin() + 1501);
        //cu_utility::printVector(sliced_vec, 10);

        cudaDeviceSynchronize();
        cudaMemcpy(dBiases_tOutput2.data(), d_biasOutput2, f_size * 300, cudaMemcpyDeviceToHost);
        cudaMemcpy(dWeights_tFlattened2.data(), d_weightOutput2, f_size * 90000, cudaMemcpyDeviceToHost);

        //cu_utility::printVector(dBiases_tOutput2, 10);
        //cu_utility::printVector(dWeights_tFlattened2, 10);
		Vector sliced_vec0(dWeights_tFlattened2.begin(), dWeights_tFlattened2.begin() + 301);
    	Vector sliced_vec4(dWeights_tFlattened2.begin() + 1200, dWeights_tFlattened2.begin() + 1501);
        
        cu_utility::printVector(sliced_vec0, 10);
        cu_utility::printVector(sliced_vec4, 10);
        cudaFree(d_biasOutput);
        cudaFree(d_weightOutput);
        cudaFree(d_testLabel);
        cudaFree(d_biasOutput2);
        cudaFree(d_weightOutput2);
        cudaFree(d_test);
        
    }
    else
    {
        std::cout << "CPU Output: \n";
        for (int i = 1; i < numLayers; i++) {
            forwardZ(weights[i - 1], biases[i - 1], activations[i - 1], zs[i]);
            sigmoid(zs[i], activations[i]);
        }
        // Run full forward z pass then run one layer of backwards
        //cu_utility::printVector(zs[numLayers - 1], 10);
        Vector delta(10, 0);
        cost_derivative(activations[numLayers - 1], testLabel, delta);
        Vector z_temp = Vector(zs[numLayers - 1].size(), 0);
        d_sigmoid(zs[numLayers - 1], z_temp);
        multiply_elementwise(z_temp, delta, dBiases_tOutput);
        outer_product(dBiases_tOutput, activations[numLayers - 2],
            dWeights_tOutput);
    	//cu_utility::printVector(dWeights_tOutput[4], 10); // correct
        //cu_utility::printVector(weights[numLayers - 2][4], 10);

        activation_derivative(weights[numLayers - 2], zs[numLayers - 1], delta);
        //cu_utility::printVector(delta, 10);
        z_temp = Vector(zs[numLayers - 2].size(), 0);
        d_sigmoid(zs[numLayers - 2], z_temp);
        multiply_elementwise(z_temp, delta, dBiases_tOutput2);
        outer_product(dBiases_tOutput2, activations[1],
            dWeights_tOutput2);
        
    }
    
    //cu_utility::printVector(dBiases_tOutput, 10);
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
        break;
    }

	for (int i = 0; i < numLayers - 1; i++) {
		std::cout << "Layer " << i << ":\n";    
		std::cout << "Weights:\n";
		cu_utility::printMatrixRowGPU(d_weights[i], layers[i + 1], layers[i], 0, 10);
		std::cout << "Biases:\n";
		cu_utility::printVectorGPU(d_biases[i], layers[i + 1], 10);
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
void deallocateVector(const std::vector<float*> &vec) {
    for (int i = 0; i < vec.size(); i++) {
        cudaFree(vec[i]);
    }
}

void CUNN::updateFromBatch(const float* d_batch, const int* d_labels, 
    const int batchSize, const int dataLen, const float learningRate) {

    std::vector<float*> d_ddWeights = allocate_like_weights();
    std::vector<float*> d_ddBiases = allocate_like_biases();

    // calculate individual gradiants and average them together
    std::vector<float*> d_dWeights = allocate_like_weights();
    std::vector<float*> d_dBiases = allocate_like_biases();
    std::vector<float*> d_delta = allocate_like_biases(); // delta.size = zsi.size for each layer i.e. like biases
    for (int i = 0; i < batchSize; i++) {
        backwards(d_dWeights, d_dBiases, d_delta,d_batch+i*dataLen, d_labels+i, dataLen);
        for (int j = 0; j < numLayers-1; j++) {
            //cu_utility::d_VectorAdd(d_ddWeights[j], d_dWeights[j], d_ddWeights[j], layers[j + 1] * layers[j], 1.0 / batchSize);
            //cu_utility::d_VectorAdd(d_ddBiases[j], d_dBiases[j], d_ddBiases[j], layers[j + 1], 1.0 / batchSize);
        }
    }
    deallocateVector(d_dWeights);
    deallocateVector(d_dBiases);
    deallocateVector(d_delta);
    // update the weights and biases with gradient computed above at the learning rate
    for (int j = 0; j < numLayers-1; j++) {
        //cu_utility::d_VectorAdd(d_weights[j], d_ddWeights[j], d_weights[j], layers[j + 1] * layers[j], -learningRate);
        //cu_utility::d_VectorAdd(d_biases[j], d_ddBiases[j], d_biases[j], layers[j + 1], -learningRate);
    }

    deallocateVector(d_ddWeights);
    deallocateVector(d_ddBiases);
}

void CUNN::backwards(std::vector<float*> &dWeights_output,
    std::vector<float*> &dBiases_output,std::vector<float*> &d_delta,
    const float* testData, const int* testLabel, size_t dataLen) {

    cudaMemcpy(d_activations[0], testData, dataLen * sizeof(float), cudaMemcpyDeviceToDevice);// activations[0] = testData;
    

    for (int i = 1; i < numLayers; i++) {
        int M = layers[i];
        int N = layers[i - 1];
        cu_utility::cuForwardLayerWithZs(d_weights[i - 1], d_biases[i - 1], d_activations[i - 1], d_zs[i],d_activations[i], M, N);
    }
    return;
    //Vector delta;
    for (int i = 0; i < numLayers - 1; i++) {
        if (i == 0) {
            cu_utility::cuBackwardOutputLayer(d_activations[numLayers - 1], d_activations[numLayers - 2], dBiases_output[numLayers - 2], dWeights_output[numLayers - 2],
                d_zs[numLayers - 1], d_delta[numLayers - 2], testLabel, layers[2], layers[3]);
            
        }
        else {
            cu_utility::cuBackwardRegularLayer(d_activations[numLayers - 2 - i], dBiases_output[numLayers - 2 - i], d_weights[numLayers - i - 1], dWeights_output[numLayers - 2 - i], d_zs[numLayers - i - 1], d_zs[numLayers - i],
                d_delta[numLayers - 2 - i], d_delta[numLayers - 1 - i], layers[2 - i], layers[3 - i], layers[4 - i]);
        }
    }
}

void CUNN::cost_derivative(const Vector& last_activation, const int label,
    Vector& result) {
    for (int i = 0; i < 10; i++) {
        if (i == label) {
            result[i] = -1.0f / (last_activation[i] + FLT_EPSILON);
        }
        else {
            result[i] = 1.0f / (1.0f - last_activation[i] + FLT_EPSILON);
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

void CUNN::activation_derivative(const Matrix& weightsMat, Vector& z,
    Vector& previous) {
    auto y = Vector(z.size(), 0);
    d_sigmoid(z, y);
    multiply_elementwise(y, previous, previous);

    //cu_utility::printVector(previous, 10);

    Matrix temp;
    transpose(weightsMat, temp);
    Vector result(temp.size(), 0.0f);
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

float CUNN::evaluate(const float *input, const int* labels, int numExamples)
{
    // timing
    auto start = std::chrono::high_resolution_clock::now();
    
    int numCorrect = cu_utility::cuForwardBatch(d_weights, d_biases, d_activations_batch, layers, input, labels, numExamples, batchSize);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\tdone." << std::endl;
    std::cout << "\tElapsed time: " << elapsed.count() << " seconds."
        << std::endl;
    float accuracy = (float)numCorrect / numExamples;
    std::cout << "\tAccuracy: " << accuracy << std::endl;
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

Vector& CUNN::d_sigmoid(const Vector& x, Vector& y) {
    assert(x.size() == y.size());
    for (int i = 0; i < x.size(); i++) {
        y[i] = d_sigmoid(x[i]);
    }
    return y;
}

