#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include "fast_float/fast_float.h"
#include <Eigen/Dense>
#include "load.h"

double string_to_double(std::string_view sv) {
    double num = 0;
    auto [ptr,err] = fast_float::from_chars(sv.data(), sv.data() + sv.size(), num);
    if (err != std::errc{}) { return 0; }
    return num;
}

void read_csv(std::string &filename, Eigen::MatrixXd &X, Eigen::VectorXd &Y) {
    std::vector<std::vector<double>> rows; // reserve
    std::ifstream file(filename);
    std::string line;

    
    std::vector<double> row;
    // row size is understood by one go throug h and predefiend row is used later
    // 

    while (std::getline(file, line)) {
        std::string_view l_view(line);
        std::size_t start=0;
        while(true){
            std::size_t end= l_view.find(',',start);
            if(end==std::string_view::npos){
                std::string_view token=l_view.substr(start);
                row.push_back(string_to_double(token));
                break;
            }
            std::string_view token=l_view.substr(start,end-start);
            row.push_back(string_to_double(token));
            start=end+1;
        }
        rows.push_back(std::move(row));
    }


    // write rows to matric here onli
    // can we map the vector to data (change rows to be on heap if so)

    int r = rows.size();
    int c = rows[0].size()-1;
    X.resize(r, c);
    Y.resize(r);
    for(int i = 0; i < r; i++){
        for(int j = 0; j < c; j++){
            X(i,j) = rows[i][j];
        }
        Y(i) = rows[i][c];
    }
}



