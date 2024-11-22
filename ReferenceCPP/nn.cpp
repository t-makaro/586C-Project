#include "nn.h"
#include "utility.h"

NN::NN(std::vector<int> layers) : layers(layers) {
    numLayers = layers.size();
    weights.reserve(numLayers - 1);
    biases.reserve(numLayers - 1);

    activations.reserve(numLayers);
    zs.reserve(numLayers);
    for (int i = 0; i < numLayers; i++) {
        activations.push_back(Vector(layers[i], 0.0));
        zs.push_back(Vector(layers[i], 0.0));
        if (i < numLayers - 1) {
            dWeights.push_back(Matrix(layers[i+1], Vector(layers[i], 0.0)));
            dBiases.push_back(Vector(layers[i + 1], 0.0));
        }
    }
}

NN::~NN() {}

void NN::copyWeights(const std::vector<Matrix> weights) {
  assert(weights.size() == numLayers - 1);
  for (int i = 0; i < weights.size(); i++) {
    assert(weights[i].size() == layers[i + 1]);
    assert(weights[i][0].size() == layers[i]);
  }
  this->weights = weights;
}

void NN::copyBiases(const std::vector<Vector> biases) {
  assert(biases.size() == numLayers - 1);
  for (int i = 0; i < biases.size(); i++) {
    assert(biases[i].size() == layers[i + 1]);
  }
  this->biases = biases;
}

Vector &NN::forward(const Vector &x, Vector &result) {
  activations[0] = x;
  for (int i = 1; i < numLayers; i++) {
    forwardLayer(weights[i - 1], biases[i - 1], activations[i - 1],
                 activations[i]);
  }
  result = activations[numLayers - 1];
  return result;
}

void NN::train(const Matrix trainingData, const std::vector<int> trainingLabels, const int iterations,
               const int batchSize, float learningRate) {
  for (int j = 0; j < iterations; j++){
    for (int i=0; i < trainingData.size(); i += batchSize){
      Matrix sampleData = sliceMatrix(trainingData, i, i+batchSize);
      std::vector<int> sampleLabels = sliceVector(trainingLabels, i, i+batchSize);
      updateFromBatch(sampleData, sampleLabels, learningRate);
      break;
    }
  }
}

void NN::updateFromBatch(const Matrix batch, const std::vector<int> labels, const float learningRate) {
  int batchSize = labels.size();
  assert(batchSize == batch.size());

  std::vector<Matrix> ddWeights;
  std::vector<Vector> ddBiases;

  for (int i = 0; i < numLayers-1; i++) {
      if (i < numLayers - 1) {
          ddWeights.push_back(Matrix(layers[i + 1], Vector(layers[i], 0.0)));
          ddBiases.push_back(Vector(layers[i + 1], 0.0));
      }
  }

  // calculate individual gradiants and average them together
  for (int i = 0; i < batchSize; i++){
    backwards(dWeights, dBiases, batch[i], labels[i]);
    for (int j=0; j<weights.size(); j++){
      add(ddWeights[j], dWeights[j], ddWeights[j], 1.0 / batchSize);
      add(ddBiases[j], dBiases[j], ddBiases[j], 1.0 / batchSize);
    }
  }
  // update the weights and biases with gradient computed above at the learning rate
  for (int j = 0; j < weights.size(); j++) {
      add(weights[j], ddWeights[j], weights[j], -learningRate);
      add(biases[j], ddBiases[j], biases[j], -learningRate);
  }
}



void NN::backwards(std::vector<Matrix> &dWeights_output, std::vector<Vector> &dBiases_output, 
               const Vector &testData, int testLabel){
  activations[0] = testData;
  for (int i = 1; i < numLayers; i++){
    forwardZ(weights[i - 1], biases[i - 1], activations[i - 1], zs[i]);
    sigmoid(zs[i], activations[i]);
  }
  // delta's length is implicitly changed in the activation derivative functions to size of previous layer
  Vector delta(10,0); 
  for (int i = 0; i < numLayers-1; i++){
    if (i==0){
        // Output layer
      cost_derivative(activations[numLayers-1], testLabel, delta);
    }
    else{
      //activation_derivative(weights[numLayers - 1 - i], zs[numLayers - i], delta);
        auto z = zs[numLayers - i];
        auto y = Vector(z.size(), 0);
        d_sigmoid(z, y);
        multiply_elementwise(y, delta, delta);
        Matrix temp;
        transpose(weights[numLayers-1-i], temp);
        Vector result(temp.size(), 0.0f);
        multiply(temp, delta, result);
        delta = result;
    }
    Vector z_temp = Vector(zs[numLayers - 1 - i].size(), 0);
    d_sigmoid(zs[numLayers - 1 - i], z_temp);
    multiply_elementwise(z_temp, delta, dBiases_output[numLayers-2-i]);
    outer_product(dBiases_output[numLayers - 2 - i], activations[numLayers-2-i], 
        dWeights_output[numLayers-2-i]);
  }
  return;
}

void NN::cost_derivative(const Vector &last_activation, const int label, Vector &result){
  for (int i=0; i < 10; i++){
      if (i == label){
        result[i] = -1.0f/(last_activation[i]+FLT_EPSILON);
      } else {
        result[i] = 1.0f/(1.0f-last_activation[i]+FLT_EPSILON);
      }
  }
  return;
}

void NN::activation_derivative(const Matrix &weightsMat, Vector &z, Vector &previous){
  auto y = Vector(z.size(), 0);
  d_sigmoid(z, y);
  multiply_elementwise(y, previous, previous);
  Matrix temp;
  transpose(weightsMat, temp);
  Vector result(temp.size(), 0.0f);
  multiply(temp, previous, result);
  previous = result;
}

void NN::transpose(const Matrix &a, Matrix &result){
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
void NN::testBackwardOutputLayer(bool isGPU, Vector& testData, int testLabel)
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
    if (isGPU)
    {

    }
    else
    {
        std::cout << "CPU Output: \n";
        for (int i = 1; i < numLayers; i++) {
            forwardZ(weights[i - 1], biases[i - 1], activations[i - 1], zs[i]);
            sigmoid(zs[i], activations[i]);
        }
        // Run full forward z pass then run one layer of backwards
        utility::printVector(zs[numLayers - 1], 10);
        Vector delta(10, 0);
        cost_derivative(activations[numLayers - 1], testLabel, delta);
        Vector z_temp = Vector(zs[numLayers - 1].size(), 0);
        d_sigmoid(zs[numLayers - 1], z_temp);
        multiply_elementwise(z_temp, delta, dBiases_tOutput);
        outer_product(dBiases_tOutput, activations[numLayers - 2],
            dWeights_tOutput);
        //utility::printVector(dWeights_tOutput[0], 10); // correct
        //cu_utility::printVector(weights[numLayers - 2][4], 10);

        activation_derivative(weights[numLayers - 2], zs[numLayers - 1], delta);
        //utility::printVector(delta, 10);
        z_temp = Vector(zs[numLayers - 2].size(), 0);
        d_sigmoid(zs[numLayers - 2], z_temp);
        multiply_elementwise(z_temp, delta, dBiases_tOutput2);
        // utility::printVector(dBiases_tOutput2, 10);

        outer_product(dBiases_tOutput2, activations[1],
            dWeights_tOutput2);
    	utility::printVector(dWeights_tOutput2[0], 10);
        utility::printVector(dWeights_tOutput2[4], 10);
    }
}

float NN::evaluate(const Matrix &testData, const std::vector<int> &testLabels) {
  int numCorrect = 0;

  // timing
  auto start = std::chrono::high_resolution_clock::now();

  Vector result(10, 0);
  for (int i = 0; i < testData.size(); i++) {
    Vector input = testData[i];
    Vector output = forward(input, result);

    int maxIndex = 0;
    float maxVal = 0;
    for (int j = 0; j < output.size(); j++) {
      if (output[j] > maxVal) {
        maxVal = output[j];
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
  std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;
  float accuracy = (float)numCorrect / testData.size();
  std::cout << "Accuracy: " << accuracy
            << std::endl;
  return accuracy;
}

Vector &NN::forwardLayer(const Matrix &w, const Vector &b, const Vector &a,
                         Vector &result) {
  return sigmoid(add(multiply(w, a, result), b, result));
}

Vector &NN::forwardZ(const Matrix &w, const Vector &b, const Vector &a, Vector &result){
  return add(multiply(w, a, result), b, result);
}

Vector &NN::sigmoid(Vector &x) {
  for (int i = 0; i < x.size(); i++) {
    x[i] = sigmoid(x[i]);
  }
  return x;
}
Vector &NN::sigmoid(const Vector &x, Vector &result) {
  for (int i = 0; i < x.size(); i++) {
    result[i] = sigmoid(x[i]);
  }
  return result;
}
Vector &NN::d_sigmoid(const Vector &x, Vector& y) {
    assert(x.size() == y.size());
  for (int i = 0; i < x.size(); i++) {
    y[i] = d_sigmoid(x[i]);
  }
  return y;
}

float NN::sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }
float NN::d_sigmoid(float x) {
  float xp = exp(-x);
  return xp / ((1.0 + xp)*(1.0 + xp)); 
}

Vector &NN::multiply(const Matrix &w, const Vector &x, Vector &result) {
  assert(result.size() == w.size() && w[0].size() == x.size());

  for (int i = 0; i < w.size(); i++) {
    float sum = 0;
    for (int j = 0; j < w[i].size(); j++) {
      sum += w[i][j] * x[j];
    }
    result[i] = sum;
  }

  return result;
}

Vector &NN::multiply_elementwise(const Vector &a, const Vector &b, Vector &result){
  for (int i = 0; i < a.size(); i++){
    result[i] = a[i]*b[i];
  }
  return result;
}

Matrix &NN::outer_product(const Vector &a, const Vector &b, Matrix &result){
  for (int i = 0; i <  a.size(); i++){
    for (int j = 0; j < b.size(); j++){
      result[i][j] = a[i]*b[j];
    }
  }
  return result;
}

Vector &NN::add(const Vector &x, const Vector &b, Vector &result) {
  assert(x.size() == b.size() && x.size() == result.size());

  for (int i = 0; i < x.size(); i++) {
    result[i] = x[i] + b[i];
  }
  return result;
}
Vector &NN::add(const Vector &x, const Vector &b, Vector &result, const float scale) {
  assert(x.size() == b.size() && x.size() == result.size());

  for (int i = 0; i < x.size(); i++) {
    result[i] = x[i] + b[i] * scale;
  }
  return result;
}

Matrix &NN::add(const Matrix &x, const Matrix &b, Matrix &result, const float scale) {
  assert(x.size() == b.size() && x.size() == result.size());

  for (int i = 0; i < x.size(); i++) {
    for (int j = 0; j < x[0].size(); j++){
      result[i][j] = x[i][j] + b[i][j] * scale;
    }
  }
  return result;
}

Matrix NN::sliceMatrix(const Matrix &matrix, size_t start_row, size_t end_row) {
    // Ensure end_row doesn't exceed the size of the matrix.
    end_row = std::min(end_row, matrix.size());
    return Matrix(matrix.begin() + start_row, matrix.begin() + end_row);
}

std::vector<int> NN::sliceVector(const std::vector<int> &vec, size_t start_index, size_t end_index) {
    // Ensure end_index doesn't exceed the size of the vector.
    end_index = std::min(end_index, vec.size());
    return std::vector<int>(vec.begin() + start_index, vec.begin() + end_index);
}