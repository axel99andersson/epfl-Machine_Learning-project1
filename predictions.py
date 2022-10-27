import numpy as np
from implementations import *
from preprocessing import *
from cross_validation import *
import csv

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def predict_feature_group(feature_nbr, degree, lambda_ ):
    y, X, id = load_test_data(feature_nbr)
    y_train, X_train, id_train = load_data(feature_nbr)

    X = remove_outliers(X)
    X_train = remove_outliers(X_train)
    
    X = standardize_data(X, -999)
    X_train = standardize_data(X_train, -999)

    X_train, X = build_entire_poly(X_train, X, degree)
    w, loss = ridge_regression(y_train, X_train, lambda_)
    prediction = (X@w)
    prediction = np.array([-1 if e > 0.5 else 1 for e in prediction])
    
    return prediction, id


prediction_0, id_0 = predict_feature_group(0, 12, 0.000151)

prediction_1, id_1 = predict_feature_group(1, 12, 4.32e-7)

prediction_2, id_2 = predict_feature_group(2, 12, 5.6234132519034906e-11)

final_id = np.concatenate((id_0, id_1, id_2))
predictions = np.concatenate((prediction_0, prediction_1, prediction_2))

create_csv_submission(final_id, predictions, "higgs_prediction_try2.csv" )