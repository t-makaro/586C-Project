#include "nn.cpp"
#include "utility.cpp"

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
  nn.evaluate(csvTrainData, trainLabels);

  // NN 3: Forward Pass Test Set
  std::cout << "Forward pass over test set..." << std::endl;
  nn.evaluate(csvTestData, testLabels);

  return 0;
}
