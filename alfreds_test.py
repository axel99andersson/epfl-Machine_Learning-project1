import numpy as np
import matplotlib.pyplot as plt
import unittest
from implementations import *
from preprocessing import *

class TestModels(unittest.TestCase):
    def setUp(self):
        self.tx = np.array([[1, 2], [3, 4]])
        self.y = np.array([1, 1])
        self.w = np.array([-1, 1])
        self.decimal_place = 5
        
    def test_least_squares_GD(self):
        w, loss = least_squares_GD(self.y, self.tx, np.array([0, 0]), 50, 0.1)
        self.assertTrue(w == self.w, "least_squares_GD() doesn't work")     
    
    def test_least_squares_SGD(self):
        w, loss = least_squares_SGD(self.y, self.tx, np.array([0, 0]), 50, 0.1)
        self.assertAlmostEqual(w, self.w, self.decimal_place, "least_squares_SGD doesn't work")

    def test_least_squares(self):
        w, loss = least_squares(self.y, self.tx)
        self.assertAlmostEqual(w, self.w, self.decimal_place)
    

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.path_train_dataset = "/Users/axelandersson/Documents/Teknik/ML_epfl/ML_course/project_ML/cc4298e0-8560-4475-8bdc-9e618184f064_epfml-project-1/train.csv"
        columns = [i for i in range(1, 32)]
        try:
            self.data = np.genfromtxt(self.path_train_dataset, delimiter=',', skip_header=1, usecols=columns)
            print("Nemas problemas!")
        except Exception as e:
            print("Problemas!", e)

    def test_normalization(self):
        stdData = standardize_data(self.data, -999)
        """
        assert(abs(stdData.mean) < 1e-4)
        assert(abs(stdData.std - 1) < 1e-4)
        assert(stdData.count(np.nan) == 0)
        """
        self.assertTrue(len(stdData[stdData > 5]) > 5)

if __name__ == "__main__":
    unittest.main()