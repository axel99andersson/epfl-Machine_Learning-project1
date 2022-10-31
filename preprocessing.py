import numpy as np
from implementations import *

"""
Contains functions to pre-process the data
"""

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
    data[data == indicator] = np.nan
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    data = (data - mean) / std
    data.astype('float')
    np.nan_to_num(data, 0)
    return data

def group_data(y, tx, id):
    """
    Groups the data into three different categories with different relevant features:
    Group 0 corresponds to PRI-jet_num = 0
    Group 1 corresponds to PRI-jet_num = 1
    Group 2 corresponds to PRI-jet_num = {2,3}
    
    Returns the labels, measurement data and indices of all three groups.
    ----------------------
    Returns: ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray
    """
    indx_0 = np.where(tx[:,22]==0)
    indx_1 = np.where(tx[:,22]==1)
    indx_2 = np.where((tx[:,22]==2))
    indx_3 = np.where((tx[:,22]==3))
    
    y_0 = y[indx_0]
    y_1 = y[indx_1]
    y_2 = y[indx_2]
    y_3 = y[indx_3]

    tx_0 = tx[indx_0]
    tx_1 = tx[indx_1]
    tx_2 = tx[indx_2]
    tx_3 = tx[indx_3]
   
    id_0 = id[indx_0]
    id_1 = id[indx_1]
    id_2 = id[indx_2]
    id_3 = id[indx_3]

    tx_0 = np.delete(tx_0,[0,4,5,6,12,22,23,24,25,26,27,28], 1)
    tx_1 = np.delete(tx_1, [3,4,5,6,12,22,26,27,28], 1)
    tx_2 = np.delete(tx_2, [22], 1) 
    tx_3 = np.delete(tx_3, [22],1 )
    return y_0, tx_0, id_0, y_1, tx_1, id_1,  y_2, tx_2, id_2, y_3, tx_3, id_3

def load_data(set_of_features):
    """
    Loads data corresponding to PRI_jet_num = set_of_features (0, 1 or 2).
    Returns the labels, measurements and indices of the chosen subset.

    ----------------------
    returns: ndarray, ndarray, ndarray
    """
    path_train_dataset = "/train.csv"
    columns = [i for i in range(2, 32)]
    tx = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=columns)
    strArr = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=1, dtype=str)
    labels = np.array([1 if myStr == 'b' else 0 for myStr in strArr])
    index = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=0)
    
    y_0, tx_0, id_0, y_1, tx_1, id_1, y_2, tx_2, id_2, y_3,tx_3,id_3 = group_data(labels,tx, index)
    if set_of_features == 0:
        return y_0, tx_0, id_0
    elif set_of_features == 1:
        return y_1, tx_1, id_1
    elif set_of_features ==2 :
        return y_2, tx_2, id_2
    else:
        return y_3, tx_3, id_3

def load_test_data(set_of_features):
    """
    Loads test data corresponding to PRI_jet_num = set_of_features (0, 1 or 2).
    Returns the labels, measurements and indices of the chosen subset.

    ----------------------
    returns: ndarray, ndarray, ndarray
    """
    path_train_dataset = "/test.csv"

    columns = [i for i in range(2, 32)]
    tx = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=columns)
    strArr = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=1, dtype=str)
    labels = np.array([1 if myStr == 'b' else 0 for myStr in strArr])
    index = np.genfromtxt(path_train_dataset, delimiter=',', skip_header=1, usecols=0)


    y_0, tx_0, id_0, y_1, tx_1, id_1, y_2, tx_2, id_2, y_3, tx_3, id_3 = group_data(labels,tx, index)
    if set_of_features == 0:
        return y_0, tx_0, id_0
    elif set_of_features == 1:
        return y_1, tx_1, id_1
    elif set_of_features ==2 :
        return y_2, tx_2, id_2
    else:
        return y_3, tx_3, id_3        

def remove_outliers(tx):
    """
    Removes outliers by method calculating inter-quartile-range and
    assigning the outliers to these values. Returns the manipulated data.

    ----------------------
    Returns: ndarray
    """

    p90, p10 = np.nanpercentile(tx, [90, 10], axis=0) 
    iqr = p90-p10
    upper_b = p10 + 1.5*iqr
    lower_b = p90 - 1.5*iqr
    for i in range(len(tx[1,:])):
        tx[np.where(tx[:, i] > upper_b[i]),i] = upper_b[i]
        tx[np.where(tx[:, i] < lower_b[i]),i] = lower_b[i]
    return tx