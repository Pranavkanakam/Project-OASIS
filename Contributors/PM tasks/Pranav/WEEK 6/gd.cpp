#include "gd.hpp"
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

void batch_gd(MatrixXd& X,VectorXd& Y,VectorXd &beta_hat,double tol,double eta, double max_iter){
    double p=X.cols();
    double n=X.rows();
    beta_hat=VectorXd::Zero(p);
    for(int i=0;i<max_iter;i++){
        VectorXd gradient = (X.transpose()*(X*beta_hat - Y))/n;
        if(gradient.norm() <tol) break;
        beta_hat -=eta * gradient;
    }
}