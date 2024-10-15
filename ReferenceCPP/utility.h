#pragma once
#include "csv-parser/single_include/csv.hpp"

using namespace std;

class utility {
    // All our utility functions should be static
private:
    /* data */
public:
    utility(/* args */);
    ~utility();
    static void PrintCSV(std::string csvPath);
    static vector<vector<float>> ReadDatasetCSV(std::string csvPath,
        std::vector<int>& results);
    static vector<float> ReadBias(std::string csvPath);
    static vector<vector<float>> ReadWeight(std::string csvPath);
};