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
  void train(const Matrix trainingData, const int iterations,
             const int batchSize, float learningRate);
  float evaluate(const Matrix &testData, const std::vector<int> &testLabels);

  void copyWeights(const std::vector<Matrix> weights);
  void copyBiases(const std::vector<Vector> biases);

private:
  int numLayers;
  std::vector<int> layers;
  std::vector<Matrix> weights;
  std::vector<Vector> biases;
  // preallocate memory for forward pass
  std::vector<Vector> activations;
  // preallocate memory for backward pass
  std::vector<Matrix> dWeights;
  std::vector<Vector> dBiases;

  static Vector &forwardLayer(const Matrix &w, const Vector &b, const Vector &a,
                              Vector &result);
  void updateFromBatch(const Matrix batch, const float learningRate);
  static Vector &multiply(const Matrix &w, const Vector &x, Vector &result);
  static Vector &add(const Vector &x, const Vector &b, Vector &result);
  static Vector &sigmoid(Vector &x);
  static float sigmoid(float x);
  static Vector &d_sigmoid(Vector &x);
  static float d_sigmoid(float x);

  void backwards(std::vector<Matrix> &dWeights_output, std::vector<Vector> &dBiases_output, 
                const Vector &testData, const int testLabel);
  static Vector cost_derivative(const Vector &last_activation, const int label);
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

void NN::train(const Matrix trainingData, const int iterations,
               const int batchSize, float learningRate) {
  // TODO
}

void NN::updateFromBatch(const Matrix batch, const float learningRate) {
  // TODO
}

void NN::backwards(std::vector<Matrix> &dWeights_output, std::vector<Vector> &dBiases_output, 
               const Vector &testData, int testLabel){
  // TODO
}

static Vector cost_derivative(const Vector &last_activation, const int label){
  Vector ans(10, 0);
  for (int i=0; i < 10; i++){
      if (i == label){
        ans[i] = -1/last_activation[i];
      } else {
        ans[i] = 1/(1-last_activation[i]);
      }
  }
  return ans;
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

Vector &NN::sigmoid(Vector &x) {
  for (int i = 0; i < x.size(); i++) {
    x[i] = sigmoid(x[i]);
  }
  return x;
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

Vector &NN::add(const Vector &x, const Vector &b, Vector &result) {
  assert(x.size() == b.size() && x.size() == result.size());

  for (int i = 0; i < x.size(); i++) {
    result[i] = x[i] + b[i];
  }
  return result;
}
