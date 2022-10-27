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
    path_train_dataset = "/Users/axelandersson/Documents/Teknik/ML_epfl/ML_course/project_ML/cc4298e0-8560-4475-8bdc-9e618184f064_epfml-project-1/train.csv"
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
    path_train_dataset = "/Users/axelandersson/Documents/Teknik/ML_epfl/ML_course/project_ML/cc4298e0-8560-4475-8bdc-9e618184f064_epfml-project-1/test.csv"
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

def create_even_data(labels, tx):
    """
    
    """
    labels_0 = np.where(labels==0)[0]
    labels_1 = np.where(labels==1)[0]
    labels_1 = labels_1[1:labels_0.shape[0]]
    labels_index = np.concatenate([labels_0, labels_1])
    labels = labels[labels_index]
    tx = tx[labels_index]
    return labels, tx

def remove_outliers(tx, toRemove = -1):
    
    p95, p5 = np.nanpercentile(tx, [97, 3], axis=0) #Set 0, 2: percentil(97,3), Set 1: percentil(98,2) 
    if toRemove == -1:
        for i in range(len(tx[1,:])):
            tx[np.where(tx[:, i] > p95[i]),i] = p95[i]
            tx[np.where(tx[:, i] < p5[i]),i] = p5[i]
    else:
        tx[np.where(tx[:, toRemove] > p95[toRemove]),toRemove] = p95[toRemove]
        tx[np.where(tx[:, toRemove] < p5[toRemove]),toRemove] = p5[toRemove]
    return tx

def remove_outliers2(tx):

    p95, p5 = np.percentile(tx, [95, 5], axis=0)
    ind0, ind1 , ind2 = [], [0,17,18,19], [0,4,5,6,12,22,23,24,25,26,27]
    ind0 = []
    print(tx.shape[1])
    print(ind1)
    if tx.shape[1] == 18:
        ind = ind0
    elif tx.shape[1] == 22:
        ind = ind1
    elif tx.shape[1] == 30:
        ind = ind2
    else:
        ind = ind2
        print("hmm")
    for i in range(len(tx[1,:])):
        tx[np.where(tx[:, i] > p95[i]),i] = p95[i]
        tx[np.where(tx[:, i] < p5[i]),i] = p5[i]

    return tx