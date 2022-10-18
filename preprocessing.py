import numpy as np


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

def destandardize_data(prediction):
    pass
