#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <iostream>
#include <vector>
#include "fast_float/fast_float.h"
#include <Eigen/Dense>

double string_to_double(std::string_view sv)
{
    double num = 0;
    auto [ptr, err] = fast_float::from_chars(sv.data(), sv.data() + sv.size(), num);
    if (err != std::errc{})
    {
        throw std::runtime_error("Failed to parse double from token: '" + std::string(sv) + "'");
    }
    return num;
}

void csv_to_matrix(std::string &filename, Eigen::MatrixXd &X, Eigen::VectorXd &Y, bool skip_header)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::string line;

    std::size_t rows = 0;
    std::size_t cols = 0;

    if (skip_header && getline(file, line))
        ; // NO OPERATION if 1st line is header.

    std::vector<double> buffer;
    buffer.reserve(500000);

    while (std::getline(file, line))
    {
        std::string_view l_view(line);

        std::size_t start = 0;
        int line_cols = 0;
        
        while (true)
        {   
            std::size_t end = l_view.find(',', start);
            if (end == std::string_view::npos)
            {
                std::string_view token = l_view.substr(start);
                buffer.push_back(string_to_double(token));
                line_cols++;
                break;
            }
            std::string_view token = l_view.substr(start, end - start);
            buffer.push_back(string_to_double(token));
            line_cols++;
            start = end + 1;
        }
        if (cols == 0)
        {
            cols = line_cols;
        }
        // else if (line_cols != cols)
        // {
        //     throw std::runtime_error("Inconsistent column count at line " + std::to_string(rows + 1) +
        //                              ": expected " + std::to_string(cols) + ", got " + std::to_string(line_cols));
        // }
        rows++;
    }

    size_t x_cols = cols - 1;

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> full_matrix(buffer.data(), rows, cols);

    X = full_matrix.leftCols(x_cols);
    Y = full_matrix.rightCols(1);
}

