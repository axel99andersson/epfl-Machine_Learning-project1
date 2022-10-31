import numpy as np

from cross_validation import *
from implementations import *
from preprocessing import *

"""
Script that does cross validation over a given range of polynomial extension degrees
and list of lambda-values for one of the subsets of data.
Prints out the best combination of degree and lambda-value.

"""
# Load Data
set_of_features = 0 #Can be either 0, 1 or 2, corresponding to PRI_jet_num
y, X, idx = load_data(set_of_features)

# Transform Data
X[X == -999] = np.nan
X = remove_outliers(X)
X = standardize_data(X, -999)

# Build Model (välj model, cross-validation för lambda och polynomgrad)
degrees = [1]
lambdas =  [3.82e-06, 0.0001225 ,6.875e-7, 6.875e-7]
lambdas = [1e-3]
#lambdas = [lambdas[set_of_features]]
#lambdas = np.linspace(1e-6, 1e-5,5)
k_fold = 5
seed = 12
k_indices = build_k_indices(y, k_fold, seed)
best_test_loss = float('inf')
best_test_loss_std = 0

best_model_accuracy = float('-inf')
best_model_accuracy_std = 0
best_lambda_degree = []
model_accuracy = 0
model_accuracy_list = []
model_train_accuracy_list = []
model_loss_list = []


for degree in degrees:
    for lambda_ in lambdas:
        print(lambda_)
        model_accuracy_list = []
        model_loss_list = []

        for k in range(k_fold):
            [loss_tr, loss_te, model_accuracy_k, model_train_accuracy_k] = cross_validation(y, X, k_indices, k, lambda_, degree)
            model_accuracy_list.append(model_accuracy_k)
            model_loss_list.append(loss_te)

        model_accuracy = np.mean(model_accuracy_list)
        model_accuracy_std = np.std(model_accuracy_list)
        
        model_loss = np.mean(model_loss_list)
        model_loss_std = np.std(model_loss_list)
        
        if model_accuracy > best_model_accuracy:
            best_model_accuracy = model_accuracy
            best_model_accuracy_std = model_accuracy_std
            best_test_loss = model_loss
            best_test_loss_std = model_loss_std
            best_lambda_degree = [lambda_, degree]

print("-------------------------")
print("Best lambda:", best_lambda_degree[0], "\nBest Degree:", best_lambda_degree[1], "\nModel Accuracy:", best_model_accuracy, "% +-", best_model_accuracy_std)
print("Test loss:", best_test_loss, "+-", best_test_loss_std)
#print("Accurracy by removed feature: ", model_accuracy)
print("-------------------------")
print("Training Done")

