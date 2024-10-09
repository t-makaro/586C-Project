#include "utility.cpp"

int main() {
    // utility::PrintCSV("../data/test.csv");

    // Util 1: Read Test Data
    // auto csvData = utility::ReadTestCSV("../data/test.csv");

    // Util 2: Read Train Data (/w labels)
    vector<int> labels;
    labels.reserve(60000);
    // auto csvTrainData = utility::ReadTrainCSV("../data/train.csv", labels);

    // for (size_t i = 0; i < labels.size(); i++)
    // {
    //     std::cout << labels[i] << std::endl;
    // }
    
    // Util 3: Read Bias for a single layer
    auto biases_a1 = utility::ReadBias("../data/biases_a1.csv");

    std::cout << "Size of this bias array: " << biases_a1.size(); << std::endl;
    for (size_t i = 0; i < biases_a1.size(); i++)
    {
        std::cout << biases_a1[i] << std::endl;
    }

    // Util 4: Read Weights as vector<vector<float>> for each layer
    
    return 0;
}