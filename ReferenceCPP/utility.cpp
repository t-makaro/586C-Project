#include "utility.h"


utility::utility(/* args */) {}

utility::~utility() {}

void utility::PrintCSV(std::string csvPath) {
  csv::CSVReader reader(csvPath);
  for (csv::CSVRow &row : reader) { // Input iterator
    for (csv::CSVField &field : row) {
      // By default, get<>() produces a std::string.
      // A more efficient get<string_view>() is also available, where the
      // resulting string_view is valid as long as the parent CSVRow is alive
      std::cout << field.get<>() << " ";
    }
    std::cout << std::endl;
  }
}

vector<vector<float>> utility::ReadDatasetCSV(std::string csvPath,
                                              std::vector<int> &results) {
  vector<vector<float>> res;
  csv::CSVReader reader(csvPath);
  for (csv::CSVRow &row : reader) { // Input iterator
    bool first_field =
        false; // First field is the label, so we don't normalize that
    vector<float> row_v;
    row_v.reserve(row.size());
    for (csv::CSVField &field : row) {
      if (!first_field) {
        results.push_back(field.get<int>()); // label
        first_field = true;
      } else {
        row_v.push_back(field.get<float>() / 255.f); // normalize
      }
    }
    res.push_back(row_v);
  }
  return res;
}

vector<float> utility::ReadBias(std::string csvPath) {
  auto res = vector<float>();

  csv::CSVReader reader(csvPath);
  res.reserve(reader.n_rows());

  for (csv::CSVRow &row : reader) { // Input iterator
    // std::cout << row.size() << std::endl;
    assert(row.size() == 1); // Bias outpus should be 1 per row, and the number
                             // of rows == layer.num_of_neurons()
    res.push_back(row[0].get<float>());
  }
  return res;
}

vector<vector<float>> utility::ReadWeight(std::string csvPath) {
  auto res = vector<vector<float>>();
  csv::CSVReader reader(csvPath);
  res.reserve(reader.n_rows());

  for (csv::CSVRow &row : reader) { // Input iterator
    vector<float> row_v;
    row_v.reserve(row.size());
    for (csv::CSVField &field : row) {
      row_v.push_back(field.get<float>());
    }
    res.push_back(row_v);
  }
  return res;
}
