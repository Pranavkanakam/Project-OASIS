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


class GLMNET {

private:
    Eigen::MatrixXf standardize_data(Eigen::Ref<Eigen::MatrixXf> X) {

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
    void revert_transformation_on(Eigen::Ref<Eigen::MatrixXf> transformation_values) {

        for (int i{1}; i<P+1; i++) {
            W(i) /= transformation_values(1, i-1);
        }

        W(0) -= transformation_values.row(0).dot(W.tail(P));

    }

    float soft_threshold(float z, float g) {

        if (g < 0) { std::cout << "invalid g value" << std::endl; return -1; }

        
        if (std::abs(z) <= g) { return 0; }
        else if (z > 0) { return z - g; }
        else { return z + g; }

    }


    // this performs coordinate-by-coordinate cyclical coordinate descent;
    void perform_CD_naive_on(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, 
        int iters_till_stop, float tolerance, float reg, float alpha, bool quiet) {

        W(0) = y.mean(); 
        /* this is just easy to do at the start, since all the other predictors are 0 mean. Further, CD doesn't update (EOL)
        the bias term
        */


        for (int i{0}; i<iters_till_stop; ++i) {

            float delta {0.0};

            for (int j{1}; j<P+1; j++) {

                Eigen::VectorXf weights_j(P);
                weights_j << W(Eigen::seq(0,j-1)) , W(Eigen::seq(j+1, P));
                

                Eigen::MatrixXf X_j(N, P);
                X_j << X(Eigen::seq(0, N-1), Eigen::seq(0, j-1)), X(Eigen::seq(0, N-1), Eigen::seq(j+1, P));
                

                Eigen::VectorXf residual = (y - (X_j * weights_j));

                float z = (((X.col(j).transpose()) * residual).value())/N;

                float updated_weight = soft_threshold(z, reg*alpha)/(1+reg*(1-alpha));
                delta += std::abs(updated_weight - W(j));

                W(j) = soft_threshold(z, reg*alpha)/(1+reg*(1-alpha));
            }

            if (delta <= tolerance && quiet == false) { 
                std::cout << "Ran for: " << i+1 << " loops before reaching tolerance." << std::endl;
                return; 
            }
        }

        if (quiet == false) { std::cout << "Ran till the end, didn't reach tolerance" << std::endl; }

    }

    // using a slightly better algorithm (as detailed in section 2.1 of the glmnet paper)
    void perform_CD_optimized1_on(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y,
    int iters_till_stop, float tolerance, float reg, float alpha, bool quiet) {

        W(0) = y.mean(); 
        Eigen::VectorXf residuals = y - (X * W);

        float delta {0.0};

        for (int i{0}; i<iters_till_stop; ++i) {
            delta = 0.0;
            

            for (int j{1}; j<P+1; ++j) {
                float z = W(j) + (X.col(j).dot(residuals))/N;

                float updated_weight = soft_threshold(z, reg*alpha)/(1 + reg*(1-alpha));
                delta += std::abs(W(j) - updated_weight);
                

                // update residuals
                residuals += (W(j) - updated_weight) * X.col(j);
        

                // finally update
                W(j) = updated_weight;
                
            }

            if (delta < tolerance && quiet == false) {
                std::cout << "Convergenced in: " << i+1 << " iterations" << std::endl;
                return;
            }

        }

        if (quiet == false) { std::cout << "Didn't cross tolerance, ran till the end" << std::endl; }
    }


    void perform_CD_covariance_on(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y,
    int iters_till_stop, float tolerance, float reg, float alpha, bool quiet) {

        std::map<std::pair<int, int>, float> data_inner_products; 
        std::map<int, float> data_response_products;

        W(0) = y.mean();


        float delta;

        for (int i{0}; i<iters_till_stop; ++i) {
            delta = 0.0;

            for (int j{1}; j<P+1; ++j) {


                if (data_response_products.find(j) == data_response_products.end()) {
                    data_response_products[j] = X.col(j).dot(y);
                }

                float z = data_response_products[j];

                for (int k{1}; k<P+1; ++k) {
                    if (W(k) != 0) {
                        std::pair<int, int> coord = (j > k) ? std::pair<int, int>{k,j} : std::pair<int, int>{j,k};

                        if (data_inner_products.find(coord) == data_inner_products.end()) {
                            data_inner_products[coord] = X.col(j).dot(X.col(k));
                        }

                        z -= data_inner_products[coord] * W(k);

                    }
                }

                z /= N;
                z += W(j);

                float updated_weight = soft_threshold(z, reg*alpha)/(1 + reg*(1-alpha));
                delta += std::abs(W(j) - updated_weight);
                W(j) = updated_weight;

            }




            if (delta < tolerance && quiet == false) { 
                std::cout << "Converged in: " << i+1 << " loops" << std::endl;
                return;
            }
            delta = 0.0;

        }


        if (quiet == false) { std::cout << "Didn't cross tolerance" << std::endl; }

    } 

public:
    int N;
    int P;

    Eigen::VectorXf W;

    // Initialzier
    GLMNET(int n, int p) {
        N = n;
        P = p;
    }

    // Initializer to also load the data while creating the GLM object.
    GLMNET(int n, int p, Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, const std::string& URL) {
        N = n;
        P = p;
        load_data_into(X, y, URL);
    } 

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
    void load_data_into(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, const std::string& URL) {
        
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

        W = Eigen::VectorXf::Random(P+1) * 0.01;

        data_file.close();
    }

    // if your data isn't standardized, this will standardize it
    void train_on(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, 
        int iters_till_stop, float tolerance, float reg, float alpha, bool quiet) {

            Eigen::MatrixXf transformation_values(2, P);
            transformation_values = standardize_data(X);

            perform_CD_covariance_on(X, y, iters_till_stop, tolerance, reg, alpha, quiet);;

            revert_transformation_on(transformation_values);


    }

    void predict_on(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> response) {
        response = X * W;
    }

};


int main() {
    
    // Setup
    const int N = 100;
    const int P = 2;

    Eigen::Matrix<float, N, P+1> X;
    Eigen::Vector<float, N> y;

    std::string URL = "/Users/kailashanand/Documents/Developer/MC_OASIS/Task_Week_7/amazing_data_0.csv";

    GLMNET glm = GLMNET(N, P, X, y, URL); // this initialzes the weights, and loads the data in one go.

    glm.train_on(X, y, 1e4, 1e-5, 0.01, 0.7, true);
    
    std::cout << glm.W << std::endl;

    return 0;
}

