#include "GLMNET.h"


Eigen::MatrixXf GLMNET::standardize_data(Eigen::Ref<Eigen::MatrixXf> X) {

    // each column of this corresponds to a predictor, first row is mean, second row is std.
    // jth column is the j+1th predictor.

    Eigen::MatrixXf transformation_values(2, P);

    // code to standardize X
    for (int j{1}; j < P + 1; ++j) {
        float mean = X.col(j).mean();
        float std = std::sqrt((X.col(j).array() - mean).square().sum() / N); // we are not dividing by N-1! We want \sum_{i} x_ij = 1 \forall j

        transformation_values(0, j - 1) = mean;
        transformation_values(1, j - 1) = std;

        X.col(j).array() -= mean;
        X.col(j).array() /= std;
    }

    return transformation_values;
}

void GLMNET::revert_transformation_on(Eigen::Ref<Eigen::MatrixXf> transformation_values) {

    for (int i{1}; i < P + 1; i++)
    {
        W(i) /= transformation_values(1, i - 1);
    }

    W(0) -= transformation_values.row(0).dot(W.tail(P));
}

float GLMNET::soft_threshold(float z, float g) {

    if (g < 0) {
        std::cout << "invalid g value" << std::endl;
        return -1;
    }

    if (std::abs(z) <= g) {
        return 0;
    } else if (z > 0) {
        return z - g;
    } else {
        return z + g;
    }
}

void GLMNET::perform_CD_naive_on(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y,
                                 int iters_till_stop, float tolerance, float reg, float alpha, bool quiet) {

    W(0) = y.mean();
    /* this is just easy to do at the start, since all the other predictors are 0 mean. Further, CD doesn't update (EOL)
    the bias term
    */

    for (int i{0}; i < iters_till_stop; ++i) {

        float delta{0.0};

        for (int j{1}; j < P + 1; j++) {

            Eigen::VectorXf weights_j(P);
            weights_j << W(Eigen::seq(0, j - 1)), W(Eigen::seq(j + 1, P));

            Eigen::MatrixXf X_j(N, P);
            X_j << X(Eigen::seq(0, N - 1), Eigen::seq(0, j - 1)), X(Eigen::seq(0, N - 1), Eigen::seq(j + 1, P));

            Eigen::VectorXf residual = (y - (X_j * weights_j));

            float z = (((X.col(j).transpose()) * residual).value()) / N;

            float updated_weight = soft_threshold(z, reg * alpha) / (1 + reg * (1 - alpha));
            delta += std::abs(updated_weight - W(j));

            W(j) = soft_threshold(z, reg * alpha) / (1 + reg * (1 - alpha));
        }

        if (delta <= tolerance && quiet == false) {
            std::cout << "Ran for: " << i + 1 << " loops before reaching tolerance." << std::endl;
            return;
        }
    }

    if (quiet == false) {
        std::cout << "Ran till the end, didn't reach tolerance" << std::endl;
    }
}


void GLMNET::perform_CD_optimized1_on(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y,
                                      int iters_till_stop, float tolerance, float reg, float alpha, bool quiet) {

    W(0) = y.mean();
    Eigen::VectorXf residuals = y - (X * W);

    float delta{0.0};

    for (int i{0}; i < iters_till_stop; ++i) {
        delta = 0.0;

        for (int j{1}; j < P + 1; ++j) {
            float z = W(j) + (X.col(j).dot(residuals)) / N;

            float updated_weight = soft_threshold(z, reg * alpha) / (1 + reg * (1 - alpha));
            delta += std::abs(W(j) - updated_weight);

            // update residuals
            residuals += (W(j) - updated_weight) * X.col(j);

            // finally update
            W(j) = updated_weight;
        }

        if (delta < tolerance && quiet == false) {
            std::cout << "Converged in: " << i + 1 << " iterations" << std::endl;
            return;
        }
    }

    if (quiet == false) {
        std::cout << "Didn't cross tolerance, ran till the end" << std::endl;
    }
}

void GLMNET::perform_CD_covariance_on(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y,
                              int iters_till_stop, float tolerance, float reg, float alpha, bool quiet) {

    std::map<std::pair<int, int>, float> data_inner_products;
    std::map<int, float> data_response_products;

    W(0) = y.mean();

    float delta;

    for (int i{0}; i < iters_till_stop; ++i) {
        delta = 0.0;

        for (int j{1}; j < P + 1; ++j) {

            if (data_response_products.find(j) == data_response_products.end()) {
                data_response_products[j] = X.col(j).dot(y);
            }

            float z = data_response_products[j];

            for (int k{1}; k < P + 1; ++k) {
                if (W(k) != 0) {
                    std::pair<int, int> coord = (j > k) ? std::pair<int, int>{k, j} : std::pair<int, int>{j, k};

                    if (data_inner_products.find(coord) == data_inner_products.end())
                    {
                        data_inner_products[coord] = X.col(j).dot(X.col(k));
                    }

                    z -= data_inner_products[coord] * W(k);
                }
            }

            z /= N;
            z += W(j);

            float updated_weight = soft_threshold(z, reg * alpha) / (1 + reg * (1 - alpha));
            delta += std::abs(W(j) - updated_weight);
            W(j) = updated_weight;
        }

        if (i % 100 == 0 && quiet == false) {
            std::cout << "Done with: " << i + 1 << "epochs" << std::endl;
        }

        if (delta < tolerance && quiet == false) {
            std::cout << "Converged in: " << i + 1 << " loops" << std::endl;
            return;
        }
        delta = 0.0;
    }

    if (quiet == false) {
        std::cout << "Didn't cross tolerance" << std::endl;
    }
}


GLMNET::GLMNET(int n, int p) {
    N = n;
    P = p;
}


GLMNET::GLMNET(int n, int p, Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, const std::string &URL) {
    N = n;
    P = p;
    GLMNET::load_data_into(X, y, URL);
}


void GLMNET::load_data_into(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, const std::string &URL) {

    std::ifstream data_file;

    // augment one column of 1s to the data (the first column)
    X.col(0) = Eigen::VectorXf::Ones(N);

    data_file.open(URL);

    if (!(data_file.is_open())) {
        std::cout << "Couldn't open file" << std::endl; // do better error handling later.
        return;
    }

    for (int i{0}; i < N; i++) {
        std::string line;
        std::getline(data_file, line);
        std::stringstream ss(line);

        for (int j{1}; j < P + 2; j++) {

            std::string predictor_val_str;
            std::getline(ss, predictor_val_str, ',');

            if (j == P + 1) {
                y(i) = std::stof(predictor_val_str) / 1.0;
            }
            else {
                X(i, j) = std::stof(predictor_val_str) / 1.0;
            }
        }
    }

    W = Eigen::VectorXf::Random(P + 1) * 0.01;

    data_file.close();
}


void GLMNET::train_on(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y,
              int iters_till_stop, float tolerance, float reg, float alpha, bool quiet) {
    std::cout << "Training in progress..." << std::endl;

    Eigen::MatrixXf transformation_values(2, P);
    transformation_values = standardize_data(X);

    perform_CD_covariance_on(X, y, iters_till_stop, tolerance, reg, alpha, quiet);
    

    revert_transformation_on(transformation_values);
}

void GLMNET::predict_on(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> response) {
    response = X * W;
}





void GLMNET::TORRENT_FC_update(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, Eigen::Ref<Eigen::VectorXi> index_set) {
    
    
    W = X(index_set, Eigen::placeholders::all).colPivHouseholderQr().solve(y(index_set));
    
}

void GLMNET::TORRENT_GD_update(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, Eigen::Ref<Eigen::VectorXi> index_set, const float& gd_step) {
    Eigen::MatrixXf X_s = X(index_set, Eigen::placeholders::all);
    Eigen::VectorXf y_s = y(index_set);

    W -= gd_step * (X_s.transpose() * (X_s * W - y_s));
}


void GLMNET::TORRENT_HYB_update(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, 
    Eigen::Ref<Eigen::VectorXi> index_set, const float& gd_step, float delta, float index_set_delta) {

    
    if (index_set_delta <= delta) {
        TORRENT_FC_update(X, y, index_set);
    } else {
        TORRENT_GD_update(X, y, index_set, gd_step);
    }

}

float GLMNET::HT_into(Eigen::Ref<Eigen::VectorXf> residuals, int k, Eigen::Ref<Eigen::VectorXi> index_set, std::vector<int>& present) {

    int same {0};

    int sorted_perm[N];
    for (int i{0}; i<N; ++i) {
        sorted_perm[i] = i;
    }

    

    std::nth_element(sorted_perm, sorted_perm + k, sorted_perm + N, [&residuals] (int a, int b) { 
        return residuals[a] < residuals[b]; }
    );


    for (int i{0}; i<N; ++i) {
        if (i<k) {
            present[sorted_perm[i]] = 1;
        } else {
            present[sorted_perm[i]] = 0;
        }
    }

    for (int i{0}; i<k; ++i) {
        same += present[index_set[i]];
    }

    
    for (int i{0}; i<k; ++i) {
        index_set[i] = sorted_perm[i];
    }

    return 1.0f - (static_cast<float>(same)/static_cast<float>(k));

}

void GLMNET::train_with_RSLR_on(Eigen::Ref<Eigen::MatrixXf> X, Eigen::Ref<Eigen::VectorXf> y, 
        int iters_till_stop, float tolerance, float g, float gd_step, float delta, bool quiet) {

    Eigen::MatrixXf transformation_values(2, P);
    transformation_values = standardize_data(X);

    int k = (1-g)*N;
    float index_set_delta {0.0};
        
    W = Eigen::VectorXf::Zero(P+1);
    Eigen::VectorXf residuals = (y).cwiseAbs();

    Eigen::VectorXi index_set(k); //iota
    std::vector<int> present(N);    
    
    HT_into(residuals, k, index_set, present); // add ols

    float prev_residual_norm = residuals(index_set).norm();

    for (int i{0}; i<iters_till_stop; ++i) {

        
        TORRENT_HYB_update(X, y, index_set, gd_step, delta, index_set_delta);
        residuals = (y - X*W).cwiseAbs(); 
        index_set_delta = HT_into(residuals, k, index_set, present);

        
        float current_residual_norm = residuals(index_set).norm();

        if ((std::abs(current_residual_norm - prev_residual_norm)) < tolerance) {
            if (quiet == false) {
                std::cout << "Converged after: " << i+1 << " iterations, with residual error: " << residuals(index_set).norm() << "\n";
            }
            quiet = true;
            break;
        }

        prev_residual_norm = current_residual_norm;

    }

    if (quiet == false) {
        std::cout << "Didn't converge, hit iteration limit. Residual error: " << residuals(index_set).norm() << "\n";
    }

    revert_transformation_on(transformation_values);

}