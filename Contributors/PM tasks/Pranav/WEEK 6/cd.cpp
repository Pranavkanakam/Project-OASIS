#include "cd.hpp"
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

void cd(MatrixXd &X,VectorXd &Y,VectorXd &beta_hat,int cycles,double diff){
    int p=X.cols();
    int n=X.rows();
    beta_hat= Eigen::VectorXd::Zero(p);
    Eigen::VectorXd r=Y;

    for(int i=0;i<cycles;i++){

        double max_diff=0;
        for(int j=0;j<p;j++){

            double zj=(X.col(j).array()*r.array()).sum()/n;
            double new_beta_j=zj;
            double old_beta_j=beta_hat(j);

            r += X.col(j) * (old_beta_j - new_beta_j);

            beta_hat[j]=new_beta_j;

            max_diff=std::max(max_diff,std::abs(new_beta_j-old_beta_j));
        }
        if(diff>max_diff){break;}
    }
}
