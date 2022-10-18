import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from preprocessing import *

path_train_dataset = "/Users/axelandersson/Documents/Teknik/ML_epfl/ML_course/project_ML/cc4298e0-8560-4475-8bdc-9e618184f064_epfml-project-1/train.csv"
columns = [i for i in range(1, 32)]

try:
    tx = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=columns)
    tx = standardize_data(tx, -999)
    print("standardize_data fungerar")
    tx[:,0] = np.ones((tx.shape[0]))
    test_tx = tx[:1000,:]
    tx = tx[1000:,:]
    strArr = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=1, dtype=str)
    labels = np.array([1 if myStr == 'b' else 0 for myStr in strArr])
    test_labels = labels[:1000]
    labels = labels[1000:]
    print("Nemas problemas!")
except Exception as e:
    print("Problemas!", e)

# Build Model
initial_w = np.random.randn(tx.shape[1])
#gamma = 1/4*np.max(np.linalg.svd(tx, full_matrices=False)[1])
gamma = 1 / 1.2e6
w, loss = reg_logistic_regression(labels, tx,1e-4, initial_w, gamma, int(100))
print("Logistisk regression är klar")

""""
# Test Data
path_test_data = "cc4298e0-8560-4475-8bdc-9e618184f064_epfml-project-1/test.csv"
columns = [i for i in range(1, 32)]
try:
    test_data = np.genfromtxt(path_test_data, delimiter=',', skip_header=1, usecols=columns)
    test_data[:,0] = np.ones((test_data.shape[0]))
    strArr = np.genfromtxt(path_test_data, delimiter=',', skip_header=1, usecols=1, dtype=str)
    test_labels = np.array([1 if myStr == 'b' else 0 for myStr in strArr])
    #print(tx)
    print(strArr)
    print("Nemas problemas!")
except Exception as e:
    print("Problemas!", e)
"""
prediction = test_tx@w
prediction = np.array([1 if e > 0 else 0 for e in prediction])
correct_predictions = np.sum(prediction == test_labels)

# for i in range(len(prediction)):
#     if prediction[i] == test_labels[i]:
#         correct_predictions += 1'

print("Model Accuracy: ", float(round(correct_predictions / len(prediction), 2))*100, "%")
print(prediction[:100])
print(test_labels[:100])

