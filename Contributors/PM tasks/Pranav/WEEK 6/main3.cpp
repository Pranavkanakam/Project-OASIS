#include "load.h"
#include "standardize.h"
#include "gd.hpp"
#include "cd.hpp"
#include "normal_eq.hpp"
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <chrono>
using namespace Eigen;
using namespace std;
using namespace chrono;

int main(){
    std::string filename="boston_housing.csv";
    std::vector<std::vector<double>>rows;
    Eigen::MatrixXd X;
    Eigen::VectorXd Y;
    file_to_rows(filename,rows);
    rows_to_matrix(X,Y,rows);

    Eigen::VectorXd means, stddev;
    standardize_train(X, means, stddev);
    standardize_val(X, means, stddev);
    double mean=Y.mean();
    Y.array()-=mean;

    Eigen::VectorXd beta_hat_gd,beta_hat_normal,beta_hat_cd;

    auto start_gd=high_resolution_clock::now();
    double eta=0.01,max_iter=10000,tol=1e-5;
    batch_gd(X,Y,beta_hat_gd,tol,eta,max_iter);
    auto end_gd=high_resolution_clock::now();
    auto duration_gd=duration_cast<microseconds>(end_gd-start_gd);

    auto start_cd=high_resolution_clock::now();
    double cycles=1000,diff=1e-5;
    cd(X,Y,beta_hat_gd,cycles,diff);
    auto end_cd=high_resolution_clock::now();
    auto duration_cd=duration_cast<microseconds>(end_cd-start_cd);

    auto start_normal=high_resolution_clock::now();
    normal_eq(X,Y,beta_hat_normal);
    auto end_normal=high_resolution_clock::now();
    auto duration_normal=duration_cast<microseconds>(end_normal-start_normal);

    cout<<duration_gd.count()<<" "<<duration_cd.count()<<" "<<duration_normal.count();
}