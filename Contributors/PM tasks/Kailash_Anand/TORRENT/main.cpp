#include <iostream>
#include <chrono>

#include "GLMNET.h"

int main() {
    
    // Setup
    const int N = 1e5;
    const int P = 5;

    Eigen::MatrixXf X(N, P+1);
    Eigen::VectorXf y(N);

    std::string URL = "/Users/kailashanand/Documents/Developer/MC_OASIS/TORRENT/amazing_data_0.csv";

    GLMNET glm = GLMNET(N, P, X, y, URL); // this initialzes the weights, and loads the data in one go.
    
    glm.train_with_RSLR_on(X, y, 1e4, 1e0, 0.6, 1e-2, 1, false);
    // glm.train_on(X, y, 1e4, 1e-3, 0.1, 0.6, false);
    
    std::cout << glm.W << std::endl;

    return 0;
}