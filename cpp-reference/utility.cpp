#include "csv-parser/single_include/csv.hpp"

using namespace std;

class utility
{
    // All our utility functions should be static
private:
    /* data */
public:
    utility(/* args */);
    ~utility();
    static void PrintCSV(std::string csvPath);
    static vector<vector<float>> ReadTestCSV(std::string csvPath);
    static vector<vector<float>> ReadTrainCSV(std::string csvPath, std::vector<int> &results); // Train csv comes with a label as the first field in a row
};

utility::utility(/* args */)
{
}

utility::~utility()
{
}

void utility::PrintCSV(std::string csvPath){
    csv::CSVReader reader(csvPath);
    for (csv::CSVRow& row: reader) { // Input iterator
        for (csv::CSVField& field: row) {
            // By default, get<>() produces a std::string.
            // A more efficient get<string_view>() is also available, where the resulting
            // string_view is valid as long as the parent CSVRow is alive
            std::cout << field.get<>() << " ";
        }
    std::cout << std::endl;
}

}


vector<vector<float>> utility::ReadTrainCSV(std::string csvPath, std::vector<int> &results){
    vector<vector<float>> res;
    csv::CSVReader reader(csvPath);
    for (csv::CSVRow& row: reader) { // Input iterator
        bool first_field = false; // First field is the label, so we don't normalize that
        vector<float> row_v ;
        row_v.reserve(row.size());
        for (csv::CSVField& field: row) {
            if(!first_field){
                results.push_back(field.get<int>()); // label
                first_field = true;
            }
            else{
                row_v.push_back(field.get<float>() / 255.f); // normalize
            }
            
        }
        res.push_back(row_v);
    }
    return res;
}

// This will just read and normalize every field as float
vector<vector<float>> utility::ReadTestCSV(std::string csvPath){
    vector<vector<float>> res;
    csv::CSVReader reader(csvPath);
    for (csv::CSVRow& row: reader) { // Input iterator
        vector<float> row_v ;
        row_v.reserve(row.size());
        for (csv::CSVField& field: row) {
                row_v.push_back(field.get<float>() / 255.f); // normalize
        }
        res.push_back(row_v);
    }
    return res;
}
