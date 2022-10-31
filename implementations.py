import numpy as np
import datetime as time
import sys as sys
import matplotlib.pyplot as plt

"""
Different machine learning methods
"""

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        loss: the loss of the model
        w: the model parameters as numpy arrays of shape (D, ) for the last iteration of GD 
    """
    w = initial_w
    for n_iter in range(max_iters):
        w = gradient_step(y, tx, w, gamma)
    loss = compute_MSE_loss(y, tx, w)
    return w, loss
    
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """ The Stochastic Gradient Descent (SGD) algirithm
    
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        loss: the loss of the model
        w: the model parameters as numpy arrays of shape (D, ) for the last iteration of GD
    """
    w = initial_w
    for n_iter in range(max_iters):
        w = w - gamma * compute_stoch_gradient(y, tx, w)
        
    loss = compute_MSE_loss(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """
    Least squares regression using normal equations

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)

    Returns:
        loss: the loss of the model (Mean squared error)
        w: the model parameters as numpy arrays of shape=(D, )
    """
    w = np.linalg.lstsq(tx, y)[0]
    loss = compute_MSE_loss(y, tx, w)
    
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations

    Args:
        y:          numpy array of shape=(N, )
        tx:         numpy array of shape=(N,D)
        lambda_:    a scalar for penalising model complexity 

    Returns:
        loss: the loss of the model (Mean squared error)
        w: the model parameters as numpy arrays of shape=(D, )
    """
   
    lambda_prime = 2*len(y)*lambda_
    w = np.linalg.solve(tx.T@tx + lambda_prime*np.eye(tx.shape[1]), tx.T@y)
    loss = compute_MSE_loss(y, tx, w)
    
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent, y = {0, 1}

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        loss: the loss of the model
        w: the model parameters as numpy arrays of shape (D, ) for the last iteration of GD
    """
    w = initial_w

    max_iters = int(max_iters)
    gradnorms = np.zeros(max_iters)
    tolgradnorm = 5e-4
    maxtime = 3 * 60
    start_time = time.datetime.now()
    for iter in range(max_iters): 
        grad = compute_grad_log_reg(y, tx, w)
        #gradnorms[iter] = np.linalg.norm(grad)

        # if (gradnorms[iter]/gradnorms[0] < tolgradnorm) or ((time.datetime.now() - start_time).seconds > maxtime):
        #     break
        w = w - gamma * grad #/ gradnorms[iter]

    f_final = compute_log_loss(y,tx,w)
    return w, f_final

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent, y = {0,1}, with regularization term lambda_*norm(w)^2

    Args:
        y:          numpy array of shape=(N, )
        tx:         numpy array of shape=(N,D)
        lambda_:    a scalar for penalising model complexity 
        initial_w:  numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters:  a scalar denoting the total number of iterations of GD
        gamma:      a scalar denoting the stepsize
        
    Returns:
        loss: the loss of the model
        w: the model parameters as numpy arrays of shape (D, ) for the last iteration of GD 
    """
    w = initial_w
    max_iters = int(max_iters)
    for iter in range(max_iters): 
        grad = compute_grad_log_reg(y, tx, w, lambda_)
        w = w - gamma * grad

    f_final = compute_log_loss(y,tx,w,lambda_)
    return w, f_final
# -------------------------- Helper Functions ----------------------------

def gradient_step(y, tx, w, gamma): #returns new w
    grad = compute_gradient(y, tx, w)
    return w - gamma * grad#/np.linalg.norm(grad)


def compute_log_loss(y,tx,w,lambda_=0):
    """
    Compute the log-loss
     Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    tx_prod_w = np.clip(tx@w, -20, 20)
    N = y.shape[0]
    
    loss = 1/N * ( np.ones((1,N)) @ np.log(np.ones((N,1)) + np.exp(tx_prod_w)) - y.T @ tx_prod_w)

    return loss[0][0]

def compute_MSE_loss(y, tx, w):
    """Calculate the loss using MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    e = y - tx@(w)
    L = (np.linalg.norm(e)**2)/(2*len(y))
    return L

def compute_grad_log_reg(y,tx,w,lambda_=0):
    """
    Computes the gradient for logistic regression
    """
    g0 = tx.T@(sigmoid(tx@w) - y) / tx.shape[0]
    return g0 + 2*lambda_ * w


def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    gradient_L = -(1 / len(y)) * tx.T @ e
    
    return gradient_L

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    
    random_index = np.random.choice(y.shape[0],1)
    return compute_gradient(y[random_index], tx[random_index], w) 

def sigmoid(x):
    """
    Sigmoid function of x
    """
    x = np.clip(x, -20, 20)
    return 1/ (np.ones((len(x),1)) + np.exp(-x))


def compute_model_accuracy(w, test_x, test_y):
    """
    prediction = test_x @ w for ridge_regression
    prediction = sigmoid(tx @ w) for log_reg_regression
    """
    #prediction = sigmoid(test_x@w)
    prediction = (test_x@w)

    prediction = np.array([1 if e[0] > 0.5 else 0 for e in prediction])
    prediction = prediction.reshape(len(prediction),1)
    
    correct_predictions = np.sum(prediction == test_y)
    model_accuracy = float(correct_predictions / len(prediction))*100
    return model_accuracy