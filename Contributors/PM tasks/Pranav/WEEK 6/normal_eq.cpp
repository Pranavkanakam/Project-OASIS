#include <Eigen/Dense>
#include "normal_eq.hpp"
using namespace Eigen;

void normal_eq(MatrixXd &X, VectorXd &Y, VectorXd &beta_hat){
    beta_hat= (X.transpose()*X).ldlt().solve(X.transpose()*Y);
}