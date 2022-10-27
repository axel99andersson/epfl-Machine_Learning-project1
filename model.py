import numpy as np
from implementations import *
from preprocessing import *
from cross_validation import *

"""
Ändra lambdas och/eller degrees
"""
# Load Data
set_of_features = 1
y, X, idx = load_data(set_of_features)

# Transform Data
X[X == -999] = np.nan

X = remove_outliers(X)
X = standardize_data(X, -999)


#y, X = create_even_data(y, X)
#X = X[:10000]
#y = y[:10000]
# Build Model (välj model, cross-validation för lambda och polynomgrad)
ro = -1
degrees = [12]
#lambdas = np.logspace(-7.5, -7, 10)
#lambdas = np.linspace(1e-11,1,11)
#lambdas =  [5.6234132519034906e-11]# 7.91e-7# ]
lambdas = [4.32e-7]
#lambdas = [0.000151]
k_fold = 5
seed = 12
k_indices = build_k_indices(y, k_fold, seed)
best_test_loss = float('inf')
best_model_accuracy = float('-inf')
best_lambda_degree = []
model_accuracy = 0
model_accuracy_list = []
feat_count = X.shape[1]

for degree in degrees:
    for lambda_ in lambdas:
        model_accuracy = 0
        for k in range(k_fold):
            [loss_tr, loss_te, model_accuracy_k] = cross_validation(y, X, k_indices, k, lambda_, degree)
            model_accuracy += model_accuracy_k

        model_accuracy /= k_fold
        if model_accuracy > best_model_accuracy:
            best_model_accuracy = model_accuracy
            best_lambda_degree = [lambda_, degree]


# lambda_ = 4.303030303030303e-07
# degree = 12
# accuracy = []
# #feat_count = 3
# set1feats = [11]
# X0 = X 
# for i in set1feats: #range(-1, feat_count)
#     X0 = remove_outliers(X0, i)
#  #(80.77508382770183   
# X1 = standardize_data(X0, -999)
# model_accuracy = 0
# for k in range(k_fold):
#     [loss_tr, loss_te, model_accuracy_k] = cross_validation(y, X1, k_indices, k, lambda_, degree)
#     model_accuracy += model_accuracy_k
# model_accuracy /= k_fold
# #accuracy.append((model_accuracy, i))
# #Better for set 1: 2, 3, 4, 6, 12, 13, 14, 19    
# #Better for set 0:
# Plot results
print("-------------------------")
print("Best lambda:", best_lambda_degree[0], "\nBest Degree:", best_lambda_degree[1], "\nModel Accuracy:", best_model_accuracy, "%")
#print("Accurracy by removed feature: ", model_accuracy)
print("-------------------------")
print("Training Done")

