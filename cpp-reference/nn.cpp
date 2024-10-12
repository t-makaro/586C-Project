#include <cassert>
#include <vector>
#include <iostream>
#include <chrono>

typedef std::vector<float> Vector;
typedef std::vector<Vector> Matrix;

class NN {
public:
  NN(std::vector<int> layers);
  ~NN();

  Vector &forward(const Vector &x, Vector &result);
  void train(const Matrix trainingData, const Vector trainingLabels, const int iterations,
             const int batchSize, float learningRate);
  float evaluate(const Matrix &testData, const std::vector<int> &testLabels);

  void copyWeights(const std::vector<Matrix> weights);
  void copyBiases(const std::vector<Vector> biases);

  static float sigmoid(float x);
  static float d_sigmoid(float x);

protected:
  int numLayers;
  std::vector<int> layers;
  std::vector<Matrix> weights;
  std::vector<Vector> biases;
  // preallocate memory for forward pass
  std::vector<Vector> activations;
  std::vector<Vector> zs;
  // preallocate memory for backward pass
  std::vector<Matrix> dWeights;
  std::vector<Vector> dBiases;

  void updateFromBatch(const Matrix batch, const Vector labels, const float learningRate);

private:

  static Vector &multiply(const Matrix &w, const Vector &x, Vector &result);
  static Vector &add(const Vector &x, const Vector &b, Vector &result);
  Matrix &add(const Matrix &x, const Matrix &b, Matrix &result, const float scale);
  Vector &add(const Vector &x, const Vector &b, Vector &result, const float scale);
  static Vector &sigmoid(Vector &x);
  static Vector &d_sigmoid(Vector &x);
  Vector &sigmoid(const Vector &x, Vector &result);
  static Vector &forwardLayer(const Matrix &w, const Vector &b, const Vector &a,
                              Vector &result);
                              
  void backwards(std::vector<Matrix> &dWeights_output, std::vector<Vector> &dBiases_output, 
                const Vector &testData, const int testLabel);
  void cost_derivative(const Vector &last_activation, const int label, Vector &result);
  void activation_derivative(const Matrix &weights, Vector &z, Vector &previous);
  Vector &forwardZ(const Matrix &w, const Vector &b, const Vector &a, Vector &result);
  Vector &multiply_elementwise(const Vector &a, const Vector &b, Vector &result);
  Matrix &outer_product(const Vector &a, const Vector &b, Matrix &result);
  void transpose(const Matrix &a, Matrix &result);
};

NN::NN(std::vector<int> layers) : layers(layers) {
  numLayers = layers.size();
  weights.reserve(numLayers - 1);
  biases.reserve(numLayers - 1);

  activations.reserve(numLayers);
  for (int i = 0; i < numLayers; i++) {
    activations.push_back(Vector(layers[i], 0.0));
    dWeights.push_back(Matrix(layers[i], Vector(layers[i + 1], 0.0)));
    dBiases.push_back(Vector(layers[i + 1], 0.0));
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

void NN::train(const Matrix trainingData, const Vector trainingLabels, const int iterations,
               const int batchSize, float learningRate) {
  // TODO
}

void NN::updateFromBatch(const Matrix batch, const Vector labels, const float learningRate) {
  int length = labels.size();
  assert(length == batch.size());

  for (int i = 0; i < length; i++){
    backwards(dWeights, dBiases, batch[i], labels[i]);
    for (int i=0; i<weights.size(); i++){
      add(weights[i], dWeights[i], weights[i], learningRate/length);
      add(biases[i], dBiases[i], biases[i], learningRate/length);
    }
  }
}



void NN::backwards(std::vector<Matrix> &dWeights_output, std::vector<Vector> &dBiases_output, 
               const Vector &testData, int testLabel){
  activations[0] = testData;
  for (int i = 1; i < numLayers; i++){
    forwardZ(weights[i - 1], biases[i - 1], activations[i - 1], zs[i]);
    sigmoid(zs[i], activations[i]);
  }
  Vector delta;
  for (int i = 0; i < numLayers-1; i++){
    if (i==0){
      cost_derivative(activations[activations.size()-1], testLabel, delta);
    }
    else{
      activation_derivative(weights[numLayers-i], zs[numLayers-i], delta);
    }
    multiply_elementwise(d_sigmoid(zs[numLayers-1-i]), delta, dBiases_output[numLayers-1-i]);
    outer_product(activations[numLayers-2-i], dBiases_output[numLayers-1-i], dWeights_output[numLayers-1-i]);
  }
}

void NN::cost_derivative(const Vector &last_activation, const int label, Vector &result){
  for (int i=0; i < 10; i++){
      if (i == label){
        result[i] = -1/last_activation[i];
      } else {
        result[i] = 1/(1-last_activation[i]);
      }
  }
  return;
}

void NN::activation_derivative(const Matrix &weights, Vector &z, Vector &previous){
  d_sigmoid(z);
  multiply_elementwise(z, previous, previous);
  Matrix temp;
  transpose(weights, temp);
  Vector result;
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
  std::cout << "Train Accuracy: " << accuracy
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
Vector &NN::d_sigmoid(Vector &x) {
  for (int i = 0; i < x.size(); i++) {
    x[i] = d_sigmoid(x[i]);
  }
  return x;
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