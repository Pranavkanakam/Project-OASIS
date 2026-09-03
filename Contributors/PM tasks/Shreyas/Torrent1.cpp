#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

const int NUM_FEATURES = 5;

struct Dataset {
    MatrixXd X;
    VectorXd y;
    vector<int> original_indices;
};

Dataset loadCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    std::getline(file, line); // Skip header

    std::vector<std::vector<double>> X_data;
    std::vector<double> y_data;
    std::vector<int> orig_idx;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row;

        // Read index column
        std::getline(ss, value, ',');
        orig_idx.push_back(std::stoi(value));

        for (int i = 0; i < NUM_FEATURES; ++i) {
            std::getline(ss, value, ',');
            row.push_back(std::stod(value));
        }

        std::getline(ss, value, ',');
        y_data.push_back(std::stod(value));

        X_data.push_back(row);
    }

    int num_samples = static_cast<int>(X_data.size());
    MatrixXd X(num_samples, NUM_FEATURES);
    VectorXd y(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        y(i) = y_data[i];
        for (int j = 0; j < NUM_FEATURES; ++j) {
            X(i, j) = X_data[i][j];
        }
    }

    return {X, y, orig_idx};
}

VectorXd OLS(const MatrixXd& X, const VectorXd& y) {
    return X.colPivHouseholderQr().solve(y);
}

double RSS(const MatrixXd& X, const VectorXd& y, const VectorXd& beta) {
    VectorXd residuals = y - X * beta;
    return residuals.squaredNorm();
}

int main() {
    Dataset data = loadCSV("dataset.csv");
    MatrixXd X = data.X; 
    VectorXd y = data.y;
    int n = X.rows();

    double alpha = 0.5; //fraction of observations we think are outliers              
    int num_iter = 50;  //when do we wanna stop worst case          
    double tol = 1e-6;  //stop when it converges              
    int k = static_cast<int>(std::floor((1.0 - alpha) * n)); //howmany samples exactly to be kept

    VectorXd beta = VectorXd::Zero(X.cols());
    MatrixXd X_active = X; //active sets is what we will be passing to OLS to find beta
    VectorXd y_active = y;
    
    double prev_rss = 1e18; //some very large value to get started
    int counter = 0;
    vector<int> indices(n);

    while (counter < num_iter) {
        beta = OLS(X_active, y_active); 
        VectorXd residuals = (y - X * beta).cwiseAbs();
        double current_rss = residuals.squaredNorm();
        if (std::abs(prev_rss - current_rss) < tol) {
            cout << "Converged at iteration " << counter << endl;
            break;
        }
        prev_rss = current_rss;

        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&](int a, int b) {
            return residuals(a) < residuals(b);
        });

        X_active.resize(k, X.cols());
        y_active.resize(k);
        for (int i = 0; i < k; ++i) {
            int idx = indices[i];
            X_active.row(i) = X.row(idx);
            y_active(i) = y(idx);
        }

        counter++;
    }

    cout << "Final Iteration: " << counter << endl;
    cout << "Final Active RSS: " << prev_rss << endl;
    cout << "Estimated Beta:\n" << beta << "\n\n";

    /*cout << "Uncounted Indices (Outliers):\n";
    for (int i = k; i < n; ++i) {
        cout << data.original_indices[indices[i]] << (i == n - 1 ? "" : ", ");
    }
    cout << endl;*/

    return 0;
}