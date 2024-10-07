#include "csv-parser/single_include/csv.hpp"

class utility
{
private:
    /* data */
public:
    utility(/* args */);
    ~utility();
    static void PrintCSV(std::string csvPath);
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
