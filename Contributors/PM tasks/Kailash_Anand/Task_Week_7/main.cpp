#include <iostream>
#include <fstream>
#include <chrono>

#include <vector>
#include <cmath>
#include <string>
#include <set>
#include <map>

#include "Eigen/Dense"
#include "Eigen/Core"

/*
data: The matrix (of pre-defined size) that you want to load the data into
N: The number of examples in the datasat
P: The number of predictors (excluding the bias)

Convention that I am assuming:
1. The data is already standardized
2. The .csv file has no header, no SI.No column and all
3. Each row is an example
4. I will be adding 1 column of 1s (bias trick)
5. First P columns are the predictors, the last column is response

The sizes I expect:
1. X: N x P+1
2. y: N x 1
*/
void load_data_into(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, std::string URL, int N, int P) {
    
    std::ifstream data_file;


    // augment one column of 1s to the data (the first column)
    X.col(0) = Eigen::VectorXf::Ones(N);

    data_file.open(URL);

    if (!(data_file.is_open())) { 
        std::cout << "Couldn't open file" << std::endl; // do better error handling later.
        return; 
    }

    for (int i{0}; i<N; i++) {
        std::string line;
        std::getline(data_file, line);
        std::stringstream ss(line);

        for (int j{1}; j<P+2; j++) {
            
            std::string predictor_val_str;
            std::getline(ss, predictor_val_str, ',');

            if (j == P+1) {
                y(i) = std::stof(predictor_val_str)/1.0;
            } else {
                X(i,j) = std::stof(predictor_val_str)/1.0;
            }

        }

    }

    data_file.close();
}

Eigen::MatrixXf standardize_data(Eigen::Ref<Eigen::MatrixXf> X) {
    const int N = static_cast<int>(X.rows());
    const int P = static_cast<int>(X.cols()) - 1;

    // each column of this corresponds to a predictor, first row is mean, second row is std.
    // jth column is the j+1th predictor.
    Eigen::MatrixXf transformation_values(2, P);

    // code to standardize X
    for (int j{1}; j<P+1; ++j) {
        float mean = X.col(j).mean();
        float std = std::sqrt((X.col(j).array() - mean).square().sum()/N); // we are not dividing by N-1! We want \sum_{i} x_ij = 1 \forall j

        transformation_values(0, j-1) = mean;
        transformation_values(1, j-1) = std;

        X.col(j).array() -= mean;
        X.col(j).array() /= std;
    }

    return transformation_values;


}


// transformation_values should have dimension 2 x P: first row should be all the means, second row should be all the stds
// weights should be a vector of dimension P+1
void revert_transformation_on(Eigen::Ref<Eigen::VectorXf> weights, Eigen::Ref<Eigen::MatrixXf> transformation_values) {
    const int P = static_cast<int>(transformation_values.cols());

    for (int i{1}; i<P+1; i++) {
        weights(i) /= transformation_values(1, i-1);
    }

    weights(0) -= transformation_values.row(0).dot(weights.tail(P));

}

float soft_threshold(float z, float g) {

    if (g < 0) { std::cout << "invalid g value" << std::endl; return -1; }

    
    if (std::abs(z) <= g) { return 0; }
    else if (z > 0) { return z - g; }
    else { return z + g; }

}


// this performs coordinate-by-coordinate cyclical coordinate descent;
void perform_CD_naive_on(Eigen::Ref<Eigen::VectorXf> weights, Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, 
    int iters_till_stop, float tolerance, float reg, float alpha, bool quiet) {

    weights(0) = y.mean(); 
    /* this is just easy to do at the start, since all the other predictors are 0 mean. Further, CD doesn't update (EOL)
    the bias term
    */

    const int N = static_cast<int>(X.rows());
    const int P = static_cast<int>(X.cols()) - 1;


    for (int i{0}; i<iters_till_stop; ++i) {

        float delta {0.0};

        for (int j{1}; j<P+1; j++) {

            Eigen::VectorXf weights_j(P);
            weights_j << weights(Eigen::seq(0,j-1)) , weights(Eigen::seq(j+1, P));
            

            Eigen::MatrixXf X_j(N, P);
            X_j << X(Eigen::seq(0, N-1), Eigen::seq(0, j-1)), X(Eigen::seq(0, N-1), Eigen::seq(j+1, P));
            

            Eigen::VectorXf residual = (y - (X_j * weights_j));

            float z = (((X.col(j).transpose()) * residual).value())/N;

            float updated_weight = soft_threshold(z, reg*alpha)/(1+reg*(1-alpha));
            delta += std::abs(updated_weight - weights(j));

            weights(j) = soft_threshold(z, reg*alpha)/(1+reg*(1-alpha));
        }

        if (delta <= tolerance && quiet == false) { 
            std::cout << "Ran for: " << i+1 << " loops before reaching tolerance." << std::endl;
            return; 
        }
    }

    if (quiet == false) { std::cout << "Ran till the end, didn't reach tolerance" << std::endl; }

}

// using a slightly better algorithm (as detailed in section 2.1 of the glmnet paper)
void perform_CD_optimized1_on(Eigen::Ref<Eigen::VectorXf> weights, Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y,
int iters_till_stop, float tolerance, float reg, float alpha, bool quiet) {

    int N = static_cast<int>(X.rows());
    int P = static_cast<int>(X.cols()) - 1;

    weights(0) = y.mean(); // same justification as before
    Eigen::VectorXf residuals = y - (X * weights);

    float delta {0.0};

    for (int i{0}; i<iters_till_stop; ++i) {
        delta = 0.0;
        

        for (int j{1}; j<P+1; ++j) {
            float z = weights(j) + (X.col(j).dot(residuals))/N;

            float updated_weight = soft_threshold(z, reg*alpha)/(1 + reg*(1-alpha));
            delta += std::abs(weights(j) - updated_weight);
            

            // update residuals
            residuals += (weights(j) - updated_weight) * X.col(j);
    

            // finally update
            weights(j) = updated_weight;
            
        }

        if (delta < tolerance && quiet == false) {
            std::cout << "Convergenced in: " << i+1 << " iterations" << std::endl;
            return;
        }

    }

    if (quiet == false) { std::cout << "Didn't cross tolerance, ran till the end" << std::endl; }
}


void perform_CD_covariance_on(Eigen::Ref<Eigen::VectorXf> weights, Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y,
int iters_till_stop, float tolerance, float reg, float alpha, bool quiet) {
    int N = static_cast<int>(X.rows());
    int P = static_cast<int>(X.cols()) - 1;


    std::map<std::pair<int, int>, float> data_inner_products; 
    std::map<int, float> data_response_products;

    weights(0) = y.mean();


    float delta;

    for (int i{0}; i<iters_till_stop; ++i) {
        delta = 0.0;

        for (int j{1}; j<P+1; ++j) {


            if (data_response_products.find(j) == data_response_products.end()) {
                data_response_products[j] = X.col(j).dot(y);
            }

            float z = data_response_products[j];

            for (int k{1}; k<P+1; ++k) {
                if (weights(k) != 0) {
                    std::pair<int, int> coord = (j > k) ? std::pair<int, int>{k,j} : std::pair<int, int>{j,k};

                    if (data_inner_products.find(coord) == data_inner_products.end()) {
                        data_inner_products[coord] = X.col(j).dot(X.col(k));
                    }

                    z -= data_inner_products[coord] * weights(k);

                }
            }

            z /= N;
            z += weights(j);

            float updated_weight = soft_threshold(z, reg*alpha)/(1 + reg*(1-alpha));
            delta += std::abs(weights(j) - updated_weight);
            weights(j) = updated_weight;

        }




        if (delta < tolerance && quiet == false) { 
            std::cout << "Converged in: " << i+1 << " loops" << std::endl;
            return;
        }
        delta = 0.0;

    }


    if (quiet == false) { std::cout << "Didn't cross tolerance" << std::endl; }

} 


void print_algo_time_delta(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, Eigen::Ref<Eigen::MatrixXf> transformation_values,
int iters_till_stop, float tolerance, float reg, float alpha) {

    const int P = static_cast<int>(X.cols()) - 1;
    Eigen::VectorXf W_1 = Eigen::VectorXf::Random(P+1) * 0.01;
    Eigen::VectorXf W_2 = Eigen::VectorXf::Random(P+1) * 0.01;
    Eigen::VectorXf W_3 = Eigen::VectorXf::Random(P+1) * 0.01;
    

    // timing the naive algorithm
    auto start = std::chrono::high_resolution_clock::now();
    
    perform_CD_naive_on(W_1, X, y, iters_till_stop, tolerance, reg, alpha, true);
    revert_transformation_on(W_2, transformation_values);

    auto end = std::chrono::high_resolution_clock::now();
    auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    std::cout << "Naive algorithm: " << time_elapsed.count() << " milliseconds" << std::endl;

    
    start = std::chrono::high_resolution_clock::now();
    
    perform_CD_optimized1_on(W_2, X, y, iters_till_stop, tolerance, reg, alpha, true);
    revert_transformation_on(W_2, transformation_values);

    end = std::chrono::high_resolution_clock::now();
    time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    std::cout << "Optimized algorithm: " << time_elapsed.count() << " milliseconds" << std::endl;
    

    start = std::chrono::high_resolution_clock::now();
    
    perform_CD_covariance_on(W_3, X, y, iters_till_stop, tolerance, reg, alpha, true);
    revert_transformation_on(W_3, transformation_values);

    end = std::chrono::high_resolution_clock::now();
    time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds> (end - start);
    std::cout << "Covariance algorithm: " << time_elapsed.count() << " milliseconds" << std::endl;

}




int main() {
    
    // Setup
    const int N = 100;
    const int P = 2;

    Eigen::Matrix<float, N, P+1> X;
    Eigen::Vector<float, N> y;

    std::string URL = "/Users/kailashanand/Documents/Developer/MC_OASIS/Task_Week6/amazing_data_0.csv";

    // loading data and standardizing
    load_data_into(X, y, URL, N, P);

    Eigen::MatrixXf transformation_values(2, P);
    transformation_values = standardize_data(X);

    // testing time delta using the three different algorithm on the same data set.
    // print_algo_time_delta(X, y, transformation_values, 1e4, 1e-3, 0.05, 0.7);

    Eigen::Vector<float, P+1> W = Eigen::Vector<float, P+1>::Random() * 0.01;


    perform_CD_optimized1_on(W, X, y, 1e4, 1e-5, 0.01, 0.7, false);
    
    std::cout << W << std::endl;
    std::cout << transformation_values << std::endl;
    
    std::cout << std::endl;

    revert_transformation_on(W, transformation_values);
    std::cout << W << std::endl;

    return 0;
}

