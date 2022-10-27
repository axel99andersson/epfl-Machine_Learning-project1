import numpy as np
from implementations import *
from preprocessing import *

path_train_dataset = "/Users/axelandersson/Documents/Teknik/ML_epfl/ML_course/project_ML/cc4298e0-8560-4475-8bdc-9e618184f064_epfml-project-1/train.csv"
columns = [i for i in range(1, 32)]

tx = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=columns)
strArr = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=1, dtype=str)
labels = np.array([1 if myStr == 'b' else 0 for myStr in strArr])
y_0, tx_0, y_1, tx_1, y_2, tx_2 = group_data(labels,tx)
tx_0 = standardize_data(tx_0, -999)
print("standardize_data fungerar")
tx_0[:,0] = np.ones((tx_0.shape[0]))

degree = 3

"""cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    
seed = 12
degree = 2
k_fold = 4
lambdas = [1e-2, 0.2e-3, 2.4, 1.3e-6, 0.8e1]
# split data in k fold
k = 0
rmse_tr = []
rmse_te = []

k_indices = build_k_indices(y_0, k_fold, seed)
for lambda_ in lambdas:
    total_tr_loss = 0
    total_te_loss = 0
    for k in range(k_fold):
        train_loss, test_loss = cross_validation(y_0,tx_0,k_indices,k,lambda_,degree)
        total_tr_loss += train_loss
        total_te_loss += test_loss
    rmse_tr.append(total_tr_loss/k_fold)
    rmse_te.append(total_te_loss/k_fold)

print("Test Error", rmse_te)
print("Train Error", rmse_tr)