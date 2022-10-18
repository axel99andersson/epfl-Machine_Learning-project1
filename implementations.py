from calendar import LocaleHTMLCalendar
from tkinter import W
import numpy as np
import datetime as time

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    favoritTolerans = 0.1
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        loss: the loss of the model
        w: the model parameters as numpy arrays of shape (2, ) for the last iteration of GD 
    """
    w = initial_w
    for n_iter in range(max_iters):
        if(n_iter % 100 == 0):
            if compute_MSE_loss(y, tx, w) < favoritTolerans: break
        w = gradient_step(y, tx, w, gamma)
    loss = compute_MSE_loss(y, tx, w)
    return w, loss
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ The Stochastic Gradient Descent (SGD) algirithm
    
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        loss: the loss of the model
        w: the model parameters as numpy arrays of shape (2, ) for the last iteration of GD
    """
    w = initial_w
    for n_iter in range(max_iters):
        w = w - gamma*compute_stoch_gradient(y, tx, w)
        
    loss = compute_MSE_loss(y, tx, w)
    return w, loss

def least_squares(y, tx):
    w = np.linalg.lstsq(tx, y)[0]
    loss = compute_MSE_loss(y, tx, w)
    
    return w, loss

def ridge_regression(y, tx, lambda_):
    lambda_prime = 2*len(y)*lambda_
    w = np.linalg.solve(tx.T@tx - lambda_prime*np.eye(tx.shape[1]), tx.T@y)
    loss = compute_MSE_loss(y, tx, w)
    
    return w, loss

def logistic_regression(y, tx, initial_w, gamma, max_iters):
    w = initial_w
    gradnorms = np.zeros((max_iters))
    tolgradnorm = 5e-4
    maxtime = 3 * 60
    start_time = time.datetime.now()
    for iter in range(max_iters): 
        print("w kommer här", w)
        grad = compute_grad_log_reg(y,tx,w)
        gradnorms[iter] = np.linalg.norm(grad)

        if (gradnorms[iter]/gradnorms[0] < tolgradnorm) or ((time.datetime.now() - start_time).seconds > maxtime):
            break
        """if iter != 1:
            alpha = alpha * gradnorms(iter-1)/gradnorms(iter)
        while alpha > alphamin and fhandle(xk - alpha*grad) > f - c*alpha*grad'.T *grad
            alpha = rho*alpha;
        """
        print("Gradient:", grad)
        w = w - gamma * grad / gradnorms[iter]
    f_final = compute_log_loss(y,tx,w)
    return w, f_final

def reg_logistic_regression(y, tx, lambda_, initial_w, gamma, max_iters):
    w = initial_w
    gradnorms = np.zeros((max_iters))
    tolgradnorm = 5e-4
    maxtime = 3 * 60
    start_time = time.datetime.now()
    for iter in range(max_iters): 
        #print("w kommer här", w)
        grad = compute_grad_log_reg(y, tx, w, lambda_)
        gradnorms[iter] = np.linalg.norm(grad)

        if (gradnorms[iter]/gradnorms[0] < tolgradnorm) or ((time.datetime.now() - start_time).seconds > maxtime):
            break
        """if iter != 1:
            alpha = alpha * gradnorms(iter-1)/gradnorms(iter)
        while alpha > alphamin and fhandle(xk - alpha*grad) > f - c*alpha*grad'.T *grad
            alpha = rho*alpha;
        """
        #print("Gradient:", grad)
        w = w - gamma * grad / gradnorms[iter]
    f_final = compute_log_loss(y,tx,w, lambda_)
    return w, f_final
# -------------------------- Helper Functions ----------------------------

def gradient_step(y, tx, w, gamma): #returns new w
    return w - gamma * compute_gradient(y, tx, w)


def compute_log_loss(y,tx,w,lambda_=0):
    tx_prod_w = tx@w
    N = y.shape[0]
    test0 = np.exp(-tx_prod_w)
    print("test0,", test0)
    test = np.log(np.ones((N)) + test0)
    f1 = y.T@test
    print("f1", f1)
    f2 = (np.ones((N)) - y).T @ (np.log(np.ones((N)) + np.exp(tx_prod_w)))
    print("f2", f2)
    f = f1 + f2 + w.T@w * lambda_
    print("f", f)
    return f

def compute_MSE_loss(y, tx, w):
    """Calculate the loss using MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    L = e.T@e / len(y)
    return L

def compute_grad_log_reg(y,tx,w,lambda_=0):
    tx_prod_w = tx@w
    g1 = -y / (np.ones(y.shape[0]) + np.exp(-tx_prod_w)) * np.exp(-tx_prod_w)
    g2 = (np.ones(y.shape[0]) - y) / (np.ones(y.shape[0]) + np.exp(tx_prod_w)) * np.exp(tx_prod_w)
    g0 = (g1+g2)@tx
    print("g0",g0)
    print("lambda", lambda_*2*w)
    g = g0 + lambda_ * 2*w
    return g


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