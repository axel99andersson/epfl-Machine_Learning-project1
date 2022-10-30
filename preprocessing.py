from math import isqrt
from xmlrpc.client import UNSUPPORTED_ENCODING
import numpy as np
from implementations import *

def standardize_data(data, indicator):
    """
    Removes missing data from an array, data. Returns an array with nan instead of
    indicator and an array mapping the missing data

    Transforms the data to mean(data) = 0 and std(data) = 1
    ----------------------
    param data: ndarray
    ----------------------
    return: ndarray, ...
    """
    data.astype('float')
    shape = data.shape
    data[data == indicator] = np.nan
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    data = (data - mean) / std
    data.astype('float')
    np.nan_to_num(data, 0)
    return data

# def standardize2(data, indicator):
#     data.astype('float')
#     shape = data.shape
#     data[data == indicator] = np.nan
#     min = np.min(data, axis=0)
#     max = np.max(data, axis=0)
#     data = (data - min) / (max-min)
#     data.astype('float')
#     #np.nan_to_num(data, 0)
#     return data

def group_data(y, tx, id):
    """
    Groups the data into three different categories with different relevant features:
    Group 0 corresponds to PRI-jet_num = 0
    Group 1 corresponds to PRI-jet_num = 1
    Group 2 corresponds to PRI-jet_num = {2,3}
    
    """
    indx_0 = np.where(tx[:,22]==0)
    indx_1 = np.where(tx[:,22]==1)
    indx_2 = np.where((tx[:,22]>1))
    
    y_0 = y[indx_0]
    y_1 = y[indx_1]
    y_2 = y[indx_2]

    tx_0 = tx[indx_0]
    tx_1 = tx[indx_1]
    tx_2 = tx[indx_2]
   
    id_0 = id[indx_0]
    id_1 = id[indx_1]
    id_2 = id[indx_2]

    tx_0 = np.delete(tx_0,[0,4,5,6,12,22,23,24,25,26,27,28], 1)
    tx_1 = np.delete(tx_1, [3,4,5,6,12,22,26,27,28], 1)
    tx_2 = np.delete(tx_2, [8,22], 1) 
    return y_0, tx_0, id_0, y_1, tx_1, id_1,  y_2, tx_2, id_2

def load_data(set_of_features):
    """
    set_of_features: 0, 1, 2
    """
    #"C:\Users\Alfred\'OneDrive - Lund University'\CS-433\Project1\train.csv"
    #"/Users/axelandersson/Documents/Teknik/ML_epfl/ML_course/project_ML/cc4298e0-8560-4475-8bdc-9e618184f064_epfml-project-1/train.csv"
    #"~/../../mnt/c/Users/Alfred/OneDrive\ -\ Lund\ University/CS-433/Project1/train.csv"
    #path_train_dataset = "train.csv"
    path_train_dataset = "/Users/eric/Downloads/cc4298e0-8560-4475-8bdc-9e618184f064_epfml-project-1/train.csv"
    columns = [i for i in range(2, 32)]
    tx = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=columns)
    strArr = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=1, dtype=str)
    labels = np.array([1 if myStr == 'b' else 0 for myStr in strArr])
    index = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=0)
    
    y_0, tx_0, id_0, y_1, tx_1, id_1, y_2, tx_2, id_2 = group_data(labels,tx, index)
    if set_of_features == 0:
        return y_0, tx_0, id_0
    elif set_of_features == 1:
        return y_1, tx_1, id_1
    else:
        return y_2, tx_2, id_2

def load_test_data(set_of_features):
    """
    set_of_features: 0, 1, 2
    """
    path_train_dataset = "/Users/eric/Downloads/cc4298e0-8560-4475-8bdc-9e618184f064_epfml-project-1/test.csv"

    #path_train_dataset = "/Users/axelandersson/Documents/Teknik/ML_epfl/ML_course/project_ML/cc4298e0-8560-4475-8bdc-9e618184f064_epfml-project-1/test.csv"
    columns = [i for i in range(2, 32)]
    tx = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=columns)
    strArr = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=1, dtype=str)
    labels = np.array([1 if myStr == 'b' else 0 for myStr in strArr])
    index = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=0)


    y_0, tx_0, id_0, y_1, tx_1, id_1, y_2, tx_2, id_2 = group_data(labels,tx, index)
    if set_of_features == 0:
        return y_0, tx_0, id_0
    elif set_of_features == 1:
        return y_1, tx_1, id_1
    else:
        return y_2, tx_2, id_2            

def remove_outliers(tx):
    
    p90, p10 = np.nanpercentile(tx, [90, 10], axis=0) 
    iqr = p90-p10
    upper_b = p10 + 1.5*iqr
    lower_b = p90 - 1.5*iqr
    for i in range(len(tx[1,:])):
        tx[np.where(tx[:, i] > upper_b[i]),i] = upper_b[i]
        tx[np.where(tx[:, i] < lower_b[i]),i] = lower_b[i]
    return tx