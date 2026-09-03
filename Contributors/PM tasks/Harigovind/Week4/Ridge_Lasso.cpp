#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

//We create a class so that we dont call the same functions multiple times while doing ridge and lasso

// ------------------------------------------------------------------
// 1.StandardScaler : To get the normalized version of observations
// ------------------------------------------------------------------

class StandardScaler
{
private:
    Eigen::VectorXd mean;                     // Here we are just declaring 2 variables
    Eigen::VectorXd stddev;

public:
    void fit(const Eigen::MatrixXd& X);      // Declaring 2 functions

    Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const;   // The second const is to not change the values of parameter within the function
};

void StandardScaler::fit(const Eigen::MatrixXd& X)
{
    int n_features = X.cols();

    // Compute mean of every feature (column)
    mean = X.colwise().mean();

    // Allocate space for standard deviations
    stddev.resize(n_features); //As we are adding one by one in loop

    // Compute standard deviation for each feature
    for (int i = 0; i < n_features; i++)
    {
        Eigen::VectorXd column = X.col(i);

        Eigen::VectorXd centered = column.array() - mean(i);

        double variance = centered.array().square().mean();

        stddev(i) = std::sqrt(variance);

        // Prevent division by zero
        if (stddev(i) == 0)
            stddev(i) = 1.0;
    }
}

Eigen::MatrixXd StandardScaler::transform(const Eigen::MatrixXd& X) const
{
    Eigen::MatrixXd X_scaled = X;

    for (int i = 0; i < X.cols(); i++)
    {
        X_scaled.col(i) =
            (X_scaled.col(i).array() - mean(i)) / stddev(i);
    }

    return X_scaled;
}

// -------------------------
// 2. Loading CSV File
// -------------------------

void loadCSV(
    const std::string& filename,
    Eigen::MatrixXd& X,
    Eigen::VectorXd& y
)
{
    std::ifstream file(filename);

    if(!file.is_open())
    {
        std::cout<<"Could not open file\n";
        return;
    }

    std::vector<std::vector<double>> data;

    std::string line;
    getline(file,line); // Skip header


    while(getline(file,line))
    {
        std::stringstream ss(line);

        std::vector<double> row;

        std::string value;

        while(getline(ss,value,','))
        {
            row.push_back(std::stod(value));
        }

        data.push_back(row);
    }

    int n_samples = data.size();                   // Number of rows (observations)
    int n_features = data[0].size() - 1;           //Number of columns excluding last one

    X.resize(n_samples,n_features);
    y.resize(n_samples);                          // Y is a vector of length n_samples


    for(int i=0;i<n_samples;i++)
    {
        for(int j=0;j<n_features;j++)
        { 
            X(i,j)=data[i][j];                   // X is the matrix with all observations and parameters
        }

        y(i)=data[i][n_features];                //y contains details about the dependent parameter
    }
}

// ------------------------------------------------
// 3.Splitting the data for Validation (80% - 20%)
// ------------------------------------------------

void trainValidationSplit(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,

    Eigen::MatrixXd& X_train,
    Eigen::MatrixXd& X_val,

    Eigen::VectorXd& y_train,
    Eigen::VectorXd& y_val
)
{
    int n_samples = X.rows();

    std::vector<int> indices(n_samples);

    for(int i=0;i<n_samples;i++)
        indices[i]=i;


    std::mt19937 gen(42);   //Because every time you run your code,you get the same split. 
    std::shuffle(           //This makes debugging and comparing Ridge/Lasso results much easier.
        indices.begin(),
        indices.end(),
        gen);         


    int train_size = 0.8*n_samples;


    X_train.resize(train_size,X.cols());
    X_val.resize(n_samples-train_size,X.cols());

    y_train.resize(train_size);
    y_val.resize(n_samples-train_size);


    for(int i=0;i<n_samples;i++)
    {
        int idx = indices[i];

        if(i<train_size)
        {
            X_train.row(i)=X.row(idx);
            y_train(i)=y(idx);
        }
        else
        {
            X_val.row(i-train_size)=X.row(idx);
            y_val(i-train_size)=y(idx);
        }
    }
}

// --------------------
// 4.Ridge Regression
// --------------------

class RidgeRegression
{
private:
    double lambda;                                // lambda and coeff are initialised
    Eigen::VectorXd coefficients;

public:
    RidgeRegression(double penalty);             //4 functions are declared

    void fit(const Eigen::MatrixXd& X,
             const Eigen::VectorXd& y);

    Eigen::VectorXd predict(
        const Eigen::MatrixXd& X
    ) const;

    Eigen::VectorXd getCoefficients() const;
};

//Defining each function

// Constructor
RidgeRegression::RidgeRegression(double penalty)
{
    lambda = penalty;
}


// Train the Ridge model
void RidgeRegression::fit(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y
)
{
    // Compute XᵀX
    Eigen::MatrixXd XtX = X.transpose() * X;

    // Compute Xᵀy
    Eigen::VectorXd Xty = X.transpose() * y;

    // Identity matrix
    Eigen::MatrixXd I =
        Eigen::MatrixXd::Identity(
            X.cols(),
            X.cols()
        );

    // Solve (XᵀX + λI)β = Xᵀy
    coefficients =
        (XtX + lambda * I)      // Taking inverse and solving takes more time
            .ldlt()
            .solve(Xty);
}


// Predict output
Eigen::VectorXd RidgeRegression::predict(
    const Eigen::MatrixXd& X
) const
{
    return X * coefficients;
}


// Return coefficients
Eigen::VectorXd RidgeRegression::getCoefficients() const
{
    return coefficients;
}

// --------------------
// 5.Lasso Regression
// --------------------

class LassoRegression
{
    private:

        double lambda;                               //initialising the variables
        int maxIterations;
        double tolerance;
        Eigen::VectorXd coefficients;

    public:

        // Constructor
        LassoRegression(                             //4 functions are declared
            double penalty,
            int iterations = 1000,
            double tol = 1e-6
        );

        // Train the model
        void fit(
            const Eigen::MatrixXd& X,
            const Eigen::VectorXd& y
        );

        // Predict using trained coefficients
        Eigen::VectorXd predict(
            const Eigen::MatrixXd& X
        ) const;

        // Return learned coefficients
        Eigen::VectorXd getCoefficients() const;
};

//Defining each function

// Constructor
LassoRegression::LassoRegression(
    double penalty,
    int iterations,
    double tol
)
{
    lambda = penalty;
    maxIterations = iterations;
    tolerance = tol;
}


// Train the model using Coordinate Descent
void LassoRegression::fit(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y
)
{
    // Number of features
    int n_features = X.cols();

    // Initialise all coefficients to zero
    coefficients = Eigen::VectorXd::Zero(n_features);

    // Coordinate Descent iterations
    for(int iter = 0; iter < maxIterations; iter++)
    {
        // Store coefficients before updating
        Eigen::VectorXd old_coefficients = coefficients;

        // Update one coefficient at a time
        for(int j = 0; j < n_features; j++)
        {
            // Compute prediction excluding feature j
            Eigen::VectorXd residual =
                y -
                (
                    X * coefficients
                    -
                    X.col(j) * coefficients(j)
                );

            // rho = x_j^T * residual
            double rho =
                X.col(j).dot(residual);

            // z = x_j^T * x_j
            double z =
                X.col(j).squaredNorm();

            // Soft Thresholding
            if(rho > lambda / 2.0)
            {
                coefficients(j) =
                    (2.0 * rho - lambda) /
                    (2.0 * z);
            }
            else if(rho < -lambda / 2.0)
            {
                coefficients(j) =
                    (2.0 * rho + lambda) /
                    (2.0 * z);
            }
            else
            {
                coefficients(j) = 0.0;
            }
        }

        // Check for convergence
        if((coefficients - old_coefficients).norm() < tolerance)
        {
            break;
        }
    }
}


// Predict outputs for new data
Eigen::VectorXd LassoRegression::predict(
    const Eigen::MatrixXd& X
) const
{
    return X * coefficients;
}


// Return learned coefficients
Eigen::VectorXd LassoRegression::getCoefficients() const
{
    return coefficients;
}

// --------------------------------------------------------
// 6.Computing MSE (Training and Testing/Validation Error)
// --------------------------------------------------------

double computeMSE(
    const Eigen::VectorXd& actual,
    const Eigen::VectorXd& predicted
)
{
    Eigen::VectorXd error = actual - predicted;               // Residual Error
                                                              // error.squaredNorm() gives RSS
    return error.squaredNorm() / actual.size();               // MSE = 1/n (RSS) gives training/test error

}

// ===========================
// 7.Main Function
// ===========================

int main()
{
    // Load the dataset and create 2 matrices X and Y
    Eigen::MatrixXd X;           //Initialising the matrices before them getting values
    Eigen::VectorXd y;

    loadCSV(
        "house_prices_dataset.csv",
        X,
        y
    );

    // Split into Training and Validation sets

    Eigen::MatrixXd X_train;
    Eigen::MatrixXd X_val;

    Eigen::VectorXd y_train;
    Eigen::VectorXd y_val;

    trainValidationSplit(
        X,
        y,
        X_train,
        X_val,
        y_train,
        y_val
    );

    // Standardize the features
    // IMPORTANT: Compute mean and standard deviation ONLY from the training data.
    // The validation data is transformed using the SAME mean and std to avoid data leakage.

    StandardScaler scaler;
    scaler.fit(X_train);
    X_train = scaler.transform(X_train);
    X_val = scaler.transform(X_val);

    // Candidate λ values
    // Chosen over different scales because regularization effects are logarithmic rather than linear.
    
    std::vector<double> lambdaValues =
    {
        0.01,
        0.1,
        0.5,
        1,
        2,
        5,
        10,
        20,
        50
    };


    // Store MSE values

    std::vector<double> ridgeTrainError;
    std::vector<double> ridgeValidationError;

    std::vector<double> lassoTrainError;
    std::vector<double> lassoValidationError;


    // Store coefficients for coefficient-path plots

    std::vector<Eigen::VectorXd> ridgeCoefficients;
    std::vector<Eigen::VectorXd> lassoCoefficients;

    // Train Ridge and Lasso for every λ value.

    for(double lambda : lambdaValues)
    {   
        //Ridge Regression
        RidgeRegression ridge(lambda);
        ridge.fit(X_train, y_train);

        Eigen::VectorXd ridgeTrainPrediction =
            ridge.predict(X_train);

        Eigen::VectorXd ridgeValidationPrediction =
            ridge.predict(X_val);

        ridgeTrainError.push_back(
            computeMSE(
                y_train,
                ridgeTrainPrediction
            )
        );

        ridgeValidationError.push_back(
            computeMSE(
                y_val,
                ridgeValidationPrediction
            )
        );

        ridgeCoefficients.push_back(
            ridge.getCoefficients()
        );

        // Lasso Regression
        LassoRegression lasso(lambda);
        lasso.fit(X_train, y_train);

        Eigen::VectorXd lassoTrainPrediction =
            lasso.predict(X_train);

        Eigen::VectorXd lassoValidationPrediction =
            lasso.predict(X_val);

        lassoTrainError.push_back(
            computeMSE(
                y_train,
                lassoTrainPrediction
            )
        );

        lassoValidationError.push_back(
            computeMSE(
                y_val,
                lassoValidationPrediction
            )
        );

        lassoCoefficients.push_back(
            lasso.getCoefficients()
        );
    }

    std::ofstream errorFile("errors.csv");
    errorFile
        << "lambda,"
        << "ridge_train,"
        << "ridge_validation,"
        << "lasso_train,"
        << "lasso_validation\n";

    for(size_t i = 0; i < lambdaValues.size(); i++)
    {
        errorFile
            << lambdaValues[i] << ","
            << ridgeTrainError[i] << ","
            << ridgeValidationError[i] << ","
            << lassoTrainError[i] << ","
            << lassoValidationError[i]
            << "\n";
    }

    errorFile.close();

    // Saving Ridge coefficient paths
    
    std::ofstream ridgeCoeffFile("ridge_coefficients.csv");
    ridgeCoeffFile << "lambda";                                               // Title of coloumn1

    for(int j = 0; j < ridgeCoefficients[0].size(); j++)  
    {
        ridgeCoeffFile << ",beta" << j+1;                                    // Title of remaining coloumns ( beta1,beta2,beta3...)
    }

    ridgeCoeffFile << "\n";                                                  // To go to next row 


    for(size_t i = 0; i < lambdaValues.size(); i++)                          // To fill each row
    {
        ridgeCoeffFile << lambdaValues[i];

        for(int j = 0; j < ridgeCoefficients[i].size(); j++)
        {
            ridgeCoeffFile << "," << ridgeCoefficients[i](j);
        }

        ridgeCoeffFile << "\n";
    }

    ridgeCoeffFile.close();

    // Saving Lasso coefficient paths

    std::ofstream lassoCoeffFile("lasso_coefficients.csv");
    lassoCoeffFile << "lambda";

    for(int j = 0; j < lassoCoefficients[0].size(); j++)
    {
        lassoCoeffFile << ",beta" << j+1;
    }

    lassoCoeffFile << "\n";


    for(size_t i = 0; i < lambdaValues.size(); i++)
    {
        lassoCoeffFile << lambdaValues[i];

        for(int j = 0; j < lassoCoefficients[i].size(); j++)
        {
            lassoCoeffFile << "," << lassoCoefficients[i](j);
        }

        lassoCoeffFile << "\n";
    }

    lassoCoeffFile.close();

    return 0;

}