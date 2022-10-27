import numpy as np
from implementations import *
from preprocessing import *
from cross_validation import *
from matplotlib import pyplot as plt

def stepper(y, x, test_ind, lambda_, degree, steps, plot = False):
    
    y_test, x_test = np.array([y[i] for i in test_ind]), np.array([x[i,:] for i in test_ind])
    y_train =  np.delete(y, test_ind)
    x_train = np.delete(x, test_ind, 0)
    
    x_train, x_test = build_entire_poly(x_train,x_test,degree)

    #initial_w, _ = ridge_regression(y_train,x_train,lambda_)
    initial_w = np.zeros(x_train.shape[1])

    _, loss_tr, gradnorms, weights = reg_logistic_regression(y_train, x_train, lambda_, initial_w, 1, steps)
    accuracy = []; loss = []
    for weight in weights:
        loss.append(compute_log_loss(y_test, x_test, weight, lambda_))
        accuracy.append(compute_model_accuracy(weight, x_test, y_test))

    if plot:
        plt.title(f"Lambda: {lambda_} Degree: {degree}")
        plt.subplot(1,2,1)
        plt.semilogy(gradnorms)
        plt.xlabel("Itteration")
        plt.ylabel("Gradient norm")
        plt.subplot(1,2,2)
        plt.plot(accuracy)
        plt.xlabel("Itteration")
        plt.ylabel("Accuracy")
        print("test")
        plt.show()
    

    #model_accuracy = compute_model_accuracy(w, x_test, y_test)

    return loss, accuracy, np.array(weights)

def build_poly(x, degree):
    """
    Transform feature x from [x] -> [x x^2 ... x^degree]
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
    
# Load Data
set_of_features = 2
y, X, _ = load_data(set_of_features)

# Transform Data
X = standardize_data(X, -999)
"""#y, X = create_even_data(y, X)
#X = X[:5000]
#y = y[:5000]
# Build Model (välj model, cross-validation för lambda och polynomgrad)"""
plot = False
degree = 5
lambda_ = 1e-6
steps = 500
test_ind = [i for i in range(int(len(y)/10))]

[loss, accuracy, weights] = stepper(y, X, test_ind, lambda_, degree, steps, plot)

print_ind = [i for i in range(0, steps-2, (int(steps/20)))]
print("-------------------------")
#print(" Printed indices:", print_ind)
for i in print_ind:
    print(" Weight diff:", np.linalg.norm(weights[i] - weights[i+1]))
    print(" Weight diff2:", np.linalg.norm(weights[i] - weights[i+2]), "\n")
print(" Accuracy:", accuracy[0], "\n")
print(" Error:", [int(loss[i]) for i in print_ind], "\n")



print("-------------------------")
print(" Steps:", steps, "\n Lambda:", lambda_, "\n Degree:", degree, "\n Model Accuracy:", accuracy[-1], "%")
print("-------------------------")
print("Training Done")
