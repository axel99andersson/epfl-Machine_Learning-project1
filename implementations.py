import numpy as np
import datetime as time
import sys as sys
import matplotlib.pyplot as plt
#from cross_validation import compute_model_accuracy

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    favoritTolerans = 0.1
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
        if(n_iter % 100 == 0):
            if compute_MSE_loss(y, tx, w) < favoritTolerans: break
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
        w = w - gamma*compute_stoch_gradient(y, tx, w)
        
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
    w = np.linalg.solve(tx.T@tx - lambda_prime*np.eye(tx.shape[1]), tx.T@y)
    loss = compute_MSE_loss(y, tx, w)
    
    return w, loss

def logistic_regression(y, tx, initial_w, gamma, max_iters):
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
    gradnorms = np.zeros((max_iters))
    tolgradnorm = 5e-4
    maxtime = 3 * 60
    start_time = time.datetime.now()
    for iter in range(max_iters): 
        grad = compute_grad_log_reg(y, tx, w)
        gradnorms[iter] = np.linalg.norm(grad)

        if (gradnorms[iter]/gradnorms[0] < tolgradnorm) or ((time.datetime.now() - start_time).seconds > maxtime):
            break
        w = w - gamma * grad / gradnorms[iter]

    f_final = compute_log_loss(y,tx,w)
    return w, f_final

def reg_logistic_regression(y, tx, lambda_, initial_w, gamma, max_iters):
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
    gradnorms = np.zeros(max_iters)
    weights = []
    tolgradnorm = 5e-4
    maxtime = 3 * 60
    start_time = time.datetime.now()
    xk = 0; rho = 0.65; c = 1e-1; alphamin = 1e-1; alpha = 0.5 #alphabar
    for iter in range(max_iters):
        if iter > max_iters/2:
            alphamin = 1e-4
        elif iter > max_iters/4:
            alphamin = 1e-3
        grad = compute_grad_log_reg(y, tx, w, lambda_)
        f = compute_log_loss(y, tx, w, lambda_)
        gradnorms[iter] = np.linalg.norm(grad)
        weights.append(w)
        if (gradnorms[iter]/gradnorms[0] < tolgradnorm) or ((time.datetime.now() - start_time).seconds > maxtime):
            break
        
        if iter != 0:
            alpha = alpha * gradnorms[iter-1]/gradnorms[iter]

        while alpha > alphamin and compute_log_loss(y, xk - alpha*grad, w, lambda_) > f - c*alpha*grad.T@grad :
            alpha = rho*alpha

        w = w - alpha * gamma * grad / gradnorms[iter]
    f_final = 0#compute_log_loss(y,tx,w, lambda_)

    return w, f_final, gradnorms, weights
# -------------------------- Helper Functions ----------------------------

def gradient_step(y, tx, w, gamma): #returns new w
    return w - gamma * compute_gradient(y, tx, w)


def compute_log_loss(y,tx,w,lambda_=0):
    tx_prod_w = np.clip(tx@w, -20, 20)
    N = y.shape[0]
    test0 = np.exp(-tx_prod_w)
    test = np.log(np.ones((N)) + test0)
    f1 = y.T@test
    f2 = (np.ones((N)) - y).T @ (np.log(np.ones((N)) + np.exp(tx_prod_w)))
    f = f1 + f2 + w.T@w * lambda_
    return f

def compute_MSE_loss(y, tx, w):
    """Calculate the loss using MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    L = 2*np.sqrt(e.T@e / len(y))
    return L

def compute_grad_log_reg(y,tx,w,lambda_=0):
    """
    Computes the gradient for logistic regression
    """
    g0 = (sigmoid(tx@w) - y)@tx / tx.shape[0]
    #print(w)
    # g1 = tx.T @ (-y / (np.ones(y.shape) + np.exp(-tx_prod_w)) * np.exp(-tx_prod_w))
    # g2 = tx.T @ ((np.ones(y.shape) - y) / (np.ones(y.shape) + np.exp(tx_prod_w)) * np.exp(tx_prod_w))
    # g = g1 + g2 + lambda_ * w
    # if max(np.exp(-tx_prod_w)) > sys.maxsize:
    #     g1 = -y
    # else:
    #     g1 = -y / (np.ones(y.shape[0]) + np.exp(-tx_prod_w)) * np.exp(-tx_prod_w)
    
    # if max(np.exp(tx_prod_w)) > sys.maxsize:
    #     g2 = (np.ones(y.shape[0]) - y)
    # else:
    #     g2 = (np.ones(y.shape[0]) - y) / (np.ones(y.shape[0]) + np.exp(tx_prod_w)) * np.exp(tx_prod_w)
        
    # g0 = (g1+g2)@tx
    return g0 + lambda_ * 2*w


def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx.dot(w)
    gradient_L = -(1 / len(y))*tx.T.dot(e)
    
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
    
    random_index = np.random.randint(0, len(y))
    e = y[random_index] - tx[random_index, :].T@w
    gradient = -tx[random_index, :].T@e

    return gradient

def sigmoid(arr):
    """
    Sigmoid function of arr
    """
    arr = np.clip(arr, -20, 20)
    return 1/(np.ones((len(arr)))+np.exp(-arr))


def compute_model_accuracy(w, test_x, test_y):
    """
    prediction = test_x @ w for ridge_regression
    prediction = sigmoid(tx @ w) for log_reg_regression
    """
    #prediction = sigmoid(test_x@w)
    prediction = (test_x@w)
    prediction = np.array([1 if e > 0.5 else 0 for e in prediction])
    
    correct_predictions = np.sum(prediction == test_y)
    model_accuracy = float(correct_predictions / len(prediction))*100
    return model_accuracy