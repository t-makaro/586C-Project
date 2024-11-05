#include "nn.h"
#include "utility.h"

int main() {
    //Util 0: Read Train Data
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
    std::cout << "Reading in weights..." << std::endl;
    auto biases_a1 = utility::ReadBias("../data/biases_a1_init.csv");
    auto weights_a1 = utility::ReadWeight("../data/weights_a1_init.csv");
    auto biases_a2 = utility::ReadBias("../data/biases_a2_init.csv");
    auto weights_a2 = utility::ReadWeight("../data/weights_a2_init.csv");
    auto biases_o = utility::ReadBias("../data/biases_o_init.csv");
    auto weights_o = utility::ReadWeight("../data/weights_o_init.csv");

    // NN 0: Init Neural Network
    std::cout << "Initializing NN..." << std::endl;

    std::vector<int> layers = { 784, 300, 300, 10 };
    NN nn(layers);

    // NN 1: Copy Weights and Biases
    nn.copyWeights({ weights_a1, weights_a2, weights_o });
    nn.copyBiases({ biases_a1, biases_a2, biases_o });

    // NN 2: Forward Pass Training Set

    std::cout << "Evaluate accuracy over training data" << std::endl;
    nn.evaluate(csvTestData, testLabels);

    // NN 3: Train on the training set

    std::cout << "Starting training..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    nn.train(csvTrainData, trainLabels, 1, 10, 0.1);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;

    // NN 4: Forward Pass Test Set
    std::cout << "Evaluate accuracy over test data" << std::endl;
    nn.evaluate(csvTestData, testLabels);

    return 0;
}


