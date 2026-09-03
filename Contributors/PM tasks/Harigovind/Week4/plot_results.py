import csv
import matplotlib.pyplot as plt

# Lists to store the values from errors.csv

lambda_values = []
ridge_train = []
ridge_validation = []
lasso_train = []
lasso_validation = []

# Open the CSV file

with open("errors.csv", "r") as file:

    # Create a CSV reader object
    reader = csv.reader(file)

    # Skip the header row
    next(reader)

    # Read each row of the CSV file
    for row in reader:

        lambda_values.append(float(row[0]))
        ridge_train.append(float(row[1]))
        ridge_validation.append(float(row[2]))
        lasso_train.append(float(row[3]))
        lasso_validation.append(float(row[4]))

# Plot: MSE vs Lambda

plt.figure(figsize=(10, 6))

plt.plot( lambda_values, ridge_train, marker='o', label="Ridge Training")

plt.plot( lambda_values, ridge_validation, marker='s',label="Ridge Validation")

plt.plot(lambda_values, lasso_train, marker='^', label="Lasso Training")

plt.plot(lambda_values,lasso_validation,marker='d',label="Lasso Validation")

# Use logarithmic scale for lambda
plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Mean Squared Error")
plt.title("Training and Validation Error vs Lambda")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("mse_vs_lambda.png")
plt.show()

# Read ridge_coefficients.csv

lambda_values = []
ridge_coefficients = []

with open("ridge_coefficients.csv", "r") as file:

    reader = csv.reader(file)
    header = next(reader)    
    num_coefficients = len(header) - 1

    # Create one list for every coefficient
    ridge_coefficients = [[] for i in range(num_coefficients)]

    for row in reader:

        lambda_values.append(float(row[0]))

        for i in range(num_coefficients):

            ridge_coefficients[i].append(float(row[i+1]))                 # Nested list with each element being the list of beta'i' values 

# Plot Ridge Coefficient Paths

plt.figure(figsize=(10,6))

for i in range(num_coefficients):

    plt.plot(
        lambda_values,
        ridge_coefficients[i],
        label=f"Beta {i+1}"
    )

plt.xscale("log")

plt.xlabel("Lambda")
plt.ylabel("Coefficient Value")
plt.title("Ridge Coefficient Paths")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("ridge_coefficient_path.png")
plt.show()

# Read lasso_coefficients.csv

lambda_values = []
lasso_coefficients = []

with open("lasso_coefficients.csv", "r") as file:

    reader = csv.reader(file)
    header = next(reader)
    num_coefficients = len(header) - 1

    lasso_coefficients = [[] for i in range(num_coefficients)]

    for row in reader:

        lambda_values.append(float(row[0]))

        for i in range(num_coefficients):
            lasso_coefficients[i].append(float(row[i+1]))


# Plot Lasso Coefficient Paths

plt.figure(figsize=(10,6))

for i in range(num_coefficients):
    plt.plot(lambda_values,lasso_coefficients[i],label="Beta"+str(i+1))

plt.xscale("log")

plt.xlabel("Lambda")
plt.ylabel("Coefficient Value")

plt.title("Lasso Coefficient Paths")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("lasso_coefficient_path.png")
plt.show()