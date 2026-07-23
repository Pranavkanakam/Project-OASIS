#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

void batch_gd(MatrixXd& X,VectorXd& Y,VectorXd &beta_hat,double tol,double eta, double max_iter);