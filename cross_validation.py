import numpy as np
from implementations import *
from matplotlib import pyplot as plt
from preprocessing import *
"""
Ändra model ridge_regression / log_reg_regression

"""
def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """

    y_test, x_test = y[k_indices][k], x[k_indices][k]
    y_train =  np.delete(y, k_indices[k])
    x_train = np.delete(x, k_indices[k], 0)
    
    x_train, x_test = build_entire_poly(x_train,x_test,degree)
    
    # Ridge Regression or Logistic Ridge Regression
    w, loss_tr = ridge_regression(y_train,x_train,lambda_)
    #initial_w = np.zeros(x_train.shape[1])
    #w, loss_tr, gradnorms, weights = reg_logistic_regression(y_train, x_train, lambda_, initial_w, 1e-2, 600)
    
    loss_te = compute_MSE_loss(y_test, x_test, w)
    model_accuracy = compute_model_accuracy(w, x_test, y_test)
    model_train_accuracy = compute_model_accuracy(w, x_train, y_train)
    return loss_tr, loss_te, model_accuracy, model_train_accuracy

def cross_validation_one_feature(y, x, k_indices, k, lambda_, degree, feature_index):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """

    y_test, x_test = y[k_indices][k], x[k_indices][k]
    y_train =  np.delete(y, k_indices[k])
    x_train = np.delete(x, k_indices[k], 0)
    
    x_train, x_test = build_poly_one_feature(x_train,x_test,degree,feature_index)


    w, loss_tr = ridge_regression(y_train,x_train,lambda_)

    loss_te = compute_MSE_loss(y_test, x_test, w)
    model_accuracy = compute_model_accuracy(w, x_test, y_test)

    return loss_tr, loss_te, model_accuracy

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    """
    
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def build_poly(x, degree):
    """
    Transform feature x from [x] -> [x x^2 ... x^degree]

    Args: 
        x:      shape=(N,)
        degree: int

    Returns:
        2D array of shape=(N, degree)
    """
    phi = np.ones((x.shape[0], degree))
    for i in range(0, degree):
        phi[:, i] = x**(i+1)
    return phi

def build_entire_poly(tx_train, tx_test, degree):
    """
    Transform matrix X from X = [x1 x2 x3 ... xd] -> [1 x1 x1^2 x1^3 ... x1^degree x2 x2^2 ... xd^degree]
    
    Args:
        tx_train: training data (ndarray)
        tx_test:  test data (ndarray)
    
    Return: 
        x_train: transformed training data (ndarray)
        x_test: transformed test data (ndarray)
    """
    x_train = np.ones((tx_train.shape[0], 1))
    x_test = np.ones((tx_test.shape[0], 1))
    for i in range(tx_train.shape[1]):
        x_train = np.concatenate((x_train, build_poly(tx_train[:, i], degree)), axis=1)
        x_test = np.concatenate((x_test, build_poly(tx_test[:, i], degree)), axis=1)
    
    return x_train, x_test

def build_poly_one_feature(tx_train, tx_test, degree, feature_index):
    """
    ex:
    x:s are columns
    [x1 x2 x3 ... xD] -> [1 x1 x2 x2^2 x2^3 ... x2^degree x3 ... xD]

    Args:
        tx_train:       training data (ndarray)
        tx_test:        test data (ndarray)
        degree:         int (polynomial degree)
        feature_index:  int (index of feature to augment)

    Return: 
        x_train: transformed training data (ndarray)
        x_test: transformed test data (ndarray)
    """
    x_train_ones = np.ones((tx_train.shape[0], 1))
    x_test_ones = np.ones((tx_test.shape[0], 1))

    poly_train = build_poly(tx_train[:, feature_index], degree)
    poly_test = build_poly(tx_test[:, feature_index], degree)

    tx_train_left = tx_train[:, 0:feature_index]
    tx_train_right = tx_train[:, feature_index+1:]

    tx_test_left = tx_test[:, 0:feature_index]
    tx_test_right = tx_test[:, feature_index+1:]

    tx_train = np.concatenate((x_train_ones, tx_train_left, poly_train, tx_train_right), axis = 1)
    tx_test = np.concatenate((x_test_ones, tx_test_left, poly_test, tx_test_right), axis = 1)
    return tx_train, tx_test