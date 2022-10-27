import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from preprocessing import *
from cross_validation import *


path_train_dataset = "/Users/axelandersson/Documents/Teknik/ML_epfl/ML_course/project_ML/cc4298e0-8560-4475-8bdc-9e618184f064_epfml-project-1/train.csv"
columns = [i for i in range(2, 32)]
tx = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=columns)
strArr = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=1, dtype=str)
labels = np.array([1 if myStr == 'b' else 0 for myStr in strArr])
index = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=0)

tx = np.delete(tx,[0,4,5,6,12,22,23,24,25,26,27,28], 1)
X = tx[:120000,:]
X_train = tx[120000:,:]
Y = labels[:120000]
y_train = labels[120000:]

X = standardize_data(X, -999)
X_train = standardize_data(X_train, -999)

X_train, X = build_entire_poly(X_train, X, 12)
w, loss = ridge_regression(y_train, X_train, np.power(10,-3.999))

accuracy = compute_model_accuracy(w,X,Y)

print(accuracy)



