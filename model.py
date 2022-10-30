from venv import create
import numpy as np
from implementations import *
from preprocessing import *
from cross_validation import *

"""
Ändra lambdas och/eller degrees
"""
# Load Data
set_of_features = 2
y, X, idx = load_data(set_of_features)

# Transform Data
X[X == -999] = np.nan

X = remove_outliers(X)
X = standardize_data(X, -999)



# Build Model (välj model, cross-validation för lambda och polynomgrad)
degrees = [12]

lambdas =  [6.875e-7]

k_fold = 5
seed = 12
k_indices = build_k_indices(y, k_fold, seed)
best_test_loss = float('inf')
best_model_accuracy = float('-inf')
best_lambda_degree = []
model_accuracy = 0
model_accuracy_list = []
model_train_accuracy_list = []
feat_count = X.shape[1]

for degree in degrees:
    for lambda_ in lambdas:
        print(lambda_)
        model_accuracy = 0
        model_train_accuracy = 0
        for k in range(k_fold):
            [loss_tr, loss_te, model_accuracy_k, model_train_accuracy_k] = cross_validation(y, X, k_indices, k, lambda_, degree)
            model_accuracy += model_accuracy_k
            model_train_accuracy += model_train_accuracy_k

        model_accuracy /= k_fold
        model_train_accuracy_list.append(model_train_accuracy)
        model_accuracy_list.append(model_accuracy)
        if model_accuracy > best_model_accuracy:
            best_model_accuracy = model_accuracy
            best_lambda_degree = [lambda_, degree]

print("-------------------------")
print("Best lambda:", best_lambda_degree[0], "\nBest Degree:", best_lambda_degree[1], "\nModel Accuracy:", best_model_accuracy, "%")
#print("Accurracy by removed feature: ", model_accuracy)
print("-------------------------")
print("Training Done")

