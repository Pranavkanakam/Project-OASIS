#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <Eigen/Dense>

struct Dataset {
    Eigen::MatrixXd X;
    Eigen::VectorXd y;
};

Dataset loadCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::getline(file, line);

    std::vector<std::vector<double>> X_data;
    std::vector<double> y_data;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        std::getline(ss, value, ',');

        for (int i = 0; i < 13; ++i) {
            std::getline(ss, value, ',');
            row.push_back(std::stod(value));
        }

        std::getline(ss, value, ',');
        y_data.push_back(std::stod(value));

        X_data.push_back(row);
    }

    int num_samples = X_data.size();
    Eigen::MatrixXd X(num_samples, 13);
    Eigen::VectorXd y(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        y(i) = y_data[i];
        for (int j = 0; j < 13; ++j) {
            X(i, j) = X_data[i][j];
        }
    }

    return {X, y};
}

void standardizeFeatures(Eigen::MatrixXd& X) {
    int num_samples = X.rows();
    int num_features = X.cols();

    for (int j = 0; j < num_features; ++j) {
        double mean = X.col(j).mean();
        double variance = (X.col(j).array() - mean).square().sum() / (num_samples - 1);
        double std_dev = std::sqrt(variance);

        if (std_dev > 1e-9) {
            X.col(j) = (X.col(j).array() - mean) / std_dev;
        }
    }
}

double calculateRSS(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::VectorXd& weights) {
    Eigen::VectorXd residuals = y - (X * weights);
    return residuals.squaredNorm();
}

int main() {
    try {
        Dataset dataset = loadCSV("Boston.csv");
        standardizeFeatures(dataset.X);

        int total_samples = dataset.X.rows();
        int train_samples = (total_samples + 1) / 2;
        int test_samples = total_samples / 2;

        Eigen::MatrixXd X_train(train_samples, 14);
        Eigen::VectorXd y_train(train_samples);
        Eigen::MatrixXd X_test(test_samples, 14);
        Eigen::VectorXd y_test(test_samples);

        int train_idx = 0;
        int test_idx = 0;

        for (int i = 0; i < total_samples; ++i) {
            if (i % 2 == 0) {
                X_train.row(train_idx).leftCols(13) = dataset.X.row(i);
                X_train(train_idx, 13) = 1.0;
                y_train(train_idx) = dataset.y(i);
                train_idx++;
            } else {
                X_test.row(test_idx).leftCols(13) = dataset.X.row(i);
                X_test(test_idx, 13) = 1.0;
                y_test(test_idx) = dataset.y(i);
                test_idx++;
            }
        }

        double lambda = 0.0;
        double tolerance = 0.001;
        int max_iterations = 10000;

        Eigen::VectorXd weights = Eigen::VectorXd::Zero(14);
        weights(13) = y_train.mean();

        Eigen::VectorXd z = Eigen::VectorXd::Zero(13);
        for (int j = 0; j < 13; ++j) {
            z(j) = X_train.col(j).squaredNorm();
        }

        bool converged = false;
        for (int iter = 0; iter < max_iterations; ++iter) {
            double max_change = 0.0;

            for (int j = 0; j < 13; ++j) {
                double prev_beta = weights(j);
                Eigen::VectorXd y_pred = X_train * weights;
                double rho = X_train.col(j).dot(y_train - y_pred + prev_beta * X_train.col(j));

                double new_beta = 0.0;
                if (rho > lambda) {
                    new_beta = (rho - lambda) / z(j);
                } else if (rho < -lambda) {
                    new_beta = (rho + lambda) / z(j);
                } else {
                    new_beta = 0.0;
                }

                weights(j) = new_beta;
                max_change = std::max(max_change, std::abs(new_beta - prev_beta));
            }

            double prev_intercept = weights(13);
            Eigen::VectorXd y_pred_no_intercept = X_train.leftCols(13) * weights.head(13);
            weights(13) = (y_train - y_pred_no_intercept).mean();
            max_change = std::max(max_change, std::abs(weights(13) - prev_intercept));

            if (max_change < tolerance) {
                converged = true;
                break;
            }
        }

        std::vector<std::string> feature_names = {
            "crim", "zn", "indus", "chas", "nox", "rm", "age", 
            "dis", "rad", "tax", "ptratio", "black", "lstat"
        };

        std::cout << "=== Lasso Regression Coefficients (lambda = " << lambda << ") ===" << std::endl;
        for (int i = 0; i < 13; ++i) {
            std::cout << feature_names[i] << " coefficient (w_" << i+1 << "): " << weights(i) << std::endl;
        }
        std::cout << "Intercept (bias / w_0): " << weights(13) << std::endl;

        double test_rss = calculateRSS(X_test, y_test, weights);
        std::cout << "\nResidual Sum of Squares (RSS) on Testing Dataset: " << test_rss << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}