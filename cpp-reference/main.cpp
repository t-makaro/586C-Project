#include "nn.cpp"
#include "utility.cpp"
#include <chrono>

int main() {
  // Util 0: Read Train Data
  vector<int> trainLabels;
  trainLabels.reserve(60000);
  std::cout << "Reading train data..." << std::endl;
  auto csvTrainData = utility::ReadDatasetCSV("../data/train.csv", trainLabels);
  std::cout << "done." << std::endl;
  std::cout << "Training data size: " << csvTrainData.size() << "x"
            << csvTrainData[0].size() << std::endl;
  std::cout << "Training labels size: " << trainLabels.size() << std::endl;

  // Util 1: Read Test Data
  vector<int> testLabels;
  std::cout << "Reading test data..." << std::endl;
  auto csvTestData = utility::ReadDatasetCSV("../data/test.csv", testLabels);
  std::cout << "done." << std::endl;
  std::cout << "Test data size: " << csvTestData.size() << "x"
            << csvTestData[0].size() << std::endl;
  std::cout << "Test labels size: " << testLabels.size() << std::endl;

  // Util 2: Read Weights and Biases
  auto biases_a1 = utility::ReadBias("../data/biases_a1.csv");
  auto weights_a1 = utility::ReadWeight("../data/weights_a1.csv");
  auto biases_a2 = utility::ReadBias("../data/biases_a2.csv");
  auto weights_a2 = utility::ReadWeight("../data/weights_a2.csv");
  auto biases_o = utility::ReadBias("../data/biases_o.csv");
  auto weights_o = utility::ReadWeight("../data/weights_o.csv");

  // NN 0: Init Neural Network
  std::vector<int> layers = {784, 300, 300, 10};
  NN nn(layers);

  // NN 1: Copy Weights and Biases
  nn.copyWeights({weights_a1, weights_a2, weights_o});
  nn.copyBiases({biases_a1, biases_a2, biases_o});

  // NN 2: Forward Pass Training Set

  std::cout << "Forward pass over training set..." << std::endl;
  int numCorrect = 0;

  // timing
  auto start = std::chrono::high_resolution_clock::now();

  Vector result(10, 0);
  for (int i = 0; i < csvTrainData.size(); i++) {
    Vector input = csvTrainData[i];
    Vector output = nn.forward(input, result);

    int maxIndex = 0;
    float maxVal = 0;
    for (int j = 0; j < output.size(); j++) {
      if (output[j] > maxVal) {
        maxVal = output[j];
        maxIndex = j;
      }
    }
    if (maxIndex == trainLabels[i]) {
      numCorrect++;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  std::cout << "done." << std::endl;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;
  std::cout << "Train Accuracy: " << (float)numCorrect / csvTrainData.size()
            << std::endl;

  // NN 3: Forward Pass Test Set
  numCorrect = 0;
  start = std::chrono::high_resolution_clock::now();

  result = Vector(10, 0);
  for (int i = 0; i < csvTestData.size(); i++) {
    Vector input = csvTestData[i];
    Vector output = nn.forward(input, result);

    int maxIndex = 0;
    float maxVal = 0;
    for (size_t j = 0; j < output.size(); j++) {
      if (output[j] > maxVal) {
        maxVal = output[j];
        maxIndex = j;
      }
    }
    if (maxIndex == testLabels[i]) {
      numCorrect++;
    }
  }
  std::cout << "done." << std::endl;
  end = std::chrono::high_resolution_clock::now();
  elapsed = end - start;
  std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;

  std::cout << "Test Accuracy: " << (float)numCorrect / csvTestData.size()
            << std::endl;

  return 0;
}
