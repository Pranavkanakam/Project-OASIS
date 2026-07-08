#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <Eigen/Dense>
#include <chrono>

// Structure to hold our raw parsed CSV data
struct Dataset {
    Eigen::MatrixXd X; 
    Eigen::VectorXd y; 
};

// Read CSV file and parse it into Eigen structures
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

        std::getline(ss, value, ',');//iterating through first line

        for (int i = 0; i < 13; ++i) {
            std::getline(ss, value, ',');
            row.push_back(std::stod(value));
        }//for the features
        std::getline(ss, value, ',');//for the predictor value

        y_data.push_back(std::stod(value));
        X_data.push_back(row);//as X_data is a vector of vectors
    }

    // Convert std::vector to Eigen Matrix/Vector structures
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

// Function to standardize features: (x - mean) / std_dev
void standardizeFeatures(Eigen::MatrixXd& X) {
    int num_samples = X.rows(); //X.rows() returns number of rows
    int num_features = X.cols();//X.cols() returns number of columns
    for (int j = 0; j < num_features; ++j) {
        double mean = X.col(j).mean(); //mean
        double variance = (X.col(j).array() - mean).square().sum() / (num_samples - 1);//var
        double std_dev = std::sqrt(variance);//stdev
        if (std_dev > 1e-9) {
            X.col(j) = (X.col(j).array() - mean) / std_dev;
        }//error prevention
    }
}
void RSS(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::VectorXd& weights) {
    Eigen::VectorXd residuals = y - (X * weights);
    std::cout << "RSS: " << residuals.squaredNorm() << std::endl;
}

int main() {
    try {
        std::cout << "Loading dataset..." << std::endl;
        
        auto start_load = std::chrono::high_resolution_clock::now();
        Dataset dataset = loadCSV("Boston.csv");
        auto end_load = std::chrono::high_resolution_clock::now();
        
        std::cout << "Loaded " << dataset.X.rows() << " samples with " 
                  << dataset.X.cols() << " features." << std::endl;

        std::cout << "Standardizing independent variables..." << std::endl;
        
        auto start_std = std::chrono::high_resolution_clock::now();
        standardizeFeatures(dataset.X);
        auto end_std = std::chrono::high_resolution_clock::now();

        int num_samples = dataset.X.rows();
        Eigen::MatrixXd X_design(num_samples, 14); 
        X_design.leftCols(13) = dataset.X;          
        X_design.col(13) = Eigen::VectorXd::Ones(num_samples); 

        std::cout << "Minimizing RSS..." << std::endl;
        double lambda= 0.0;
        Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(14, 14);
        identity(13, 13) = 0.0;
        
        auto start_solve = std::chrono::high_resolution_clock::now();
        
        Eigen::MatrixXd XtX = X_design.transpose() * X_design+ lambda * identity;
        Eigen::VectorXd Xty = X_design.transpose() * dataset.y;

        //Eigen::VectorXd weights = XtX.colPivHouseholderQr().solve(Xty);
        Eigen::VectorXd weights = XtX.ldlt().solve(Xty);//1.554ms
        //Eigen::VectorXd weights = (XtX).inverse() * Xty; //1.365ms

        
        auto end_solve = std::chrono::high_resolution_clock::now();

        RSS(X_design, dataset.y, weights);

        auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_load - start_load).count();
        auto std_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_std - start_std).count();
        auto solve_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_solve - start_solve).count();

        std::cout << "\n=== Regression Coefficients ===" << std::endl;
        std::vector<std::string> feature_names = {
            "crim", "zn", "indus", "chas", "nox", "rm", "age", 
            "dis", "rad", "tax", "ptratio", "black", "lstat"
        };

        for (int i = 0; i < 13; ++i) {
            std::cout << feature_names[i] << " coefficient (w_" << i+1 << "): " << weights(i) << std::endl;
        }
        std::cout << "Intercept (bias / w_0): " << weights(13) << std::endl;

        std::cout << "1) Loading Time       : " << load_duration << " microseconds (" << load_duration / 1000.0 << " ms)" << std::endl;
        std::cout << "2) Standardizing Time : " << std_duration << " microseconds (" << std_duration / 1000.0 << " ms)" << std::endl;
        std::cout << "3) Solving Time       : " << solve_duration << " microseconds (" << solve_duration / 1000.0 << " ms)" << std::endl;
        std::cout << "=========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}