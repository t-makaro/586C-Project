#include "nn.cpp"
#include "utility.cpp"

int main() {
  // Util 0: Read Train Data
  vector<int> labels;
  labels.reserve(60000);
  std::cout << "Reading train data..." << std::endl;
  auto csvTrainData = utility::ReadTrainCSV("../data/train.csv", labels);
  std::cout << "done." << std::endl;
  std::cout << "Training data size: " << csvTrainData.size() << "x"
            << csvTrainData[0].size() << std::endl;
  std::cout << "Training labels size: " << labels.size() << std::endl;

  // Util 1: Read Test Data
  std::cout << "Reading test data..." << std::endl;
  auto csvTestData = utility::ReadTestCSV("../data/test.csv");
  std::cout << "done." << std::endl;
  std::cout << "Test data size: " << csvTestData.size() << "x"
            << csvTestData[0].size() << std::endl;

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

  // NN 2: Forward Pass

  std::cout << "Forward pass over training set..." << std::endl;
  int numCorrect = 0;
  Vector result(10, 0);
  for (size_t i = 0; i < csvTrainData.size(); i++) {
    auto input = csvTrainData[i];
    auto output = nn.forward(input, result);
    // check correct
    int maxIndex = 0;
    float maxVal = 0;
    for (size_t j = 0; j < output.size(); j++) {
      if (output[j] > maxVal) {
        maxVal = output[j];
        maxIndex = j;
      }
    }
    if (maxIndex == labels[i]) {
      numCorrect++;
    }
  }
  std::cout << "done." << std::endl;

  // print accuracy
  std::cout << "Accuracy: " << (float)numCorrect / csvTrainData.size()
            << std::endl;
  return 0;
}
