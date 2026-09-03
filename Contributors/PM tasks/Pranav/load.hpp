#pragma once
#include <string>
#include <string_view>
#include <vector>
#include <Eigen/Dense>
#include "load.cpp"

double string_to_double(std::string_view sv);

void csv_to_matrix(std::string &filename, Eigen::MatrixXd &X, Eigen::VectorXd &Y, bool skip_header);
