#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

void cd(MatrixXd &X,VectorXd &Y,VectorXd &beta_hat,int cycles,double diff);