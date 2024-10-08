#include "utility.cpp"

int main() {
    // utility::PrintCSV("../data/test.csv");
    auto csvData = utility::ReadTestCSV("../data/test.csv");

    vector<int> labels;
    labels.reserve(60000);
    auto csvTrainData = utility::ReadTrainCSV("../data/train.csv", labels);

    for (size_t i = 0; i < labels.size(); i++)
    {
        std::cout << labels[i] << std::endl;
    }
    

    return 0;
}