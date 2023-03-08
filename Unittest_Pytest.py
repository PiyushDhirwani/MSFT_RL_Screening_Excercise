import numpy as np
import unittest
import pytest

from DecisionTreeClassification import *

class TestDecisionTree(unittest.TestCase):

    def setUp(self):
        self.X_train = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ])
        self.y_train = np.array([0, 0, 1, 1])
        self.X_test = np.array([
            [2, 3, 4],
            [5, 6, 7],
            [8, 9, 10]
        ])
        self.y_test = np.array([0, 0, 1])
        self.model = DecisionTree(max_depth=2)
        self.model.fit(self.X_train, self.y_train)

    def test_predict(self):
        y_pred = self.model.predict(self.X_test)
        np.testing.assert_array_equal(y_pred, self.y_test)

class TestDecisionTreePytest:

    X_train = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([
        [2, 3, 4],
        [5, 6, 7],
        [8, 9, 10]
    ])
    y_test = np.array([0, 0, 1])
    model = DecisionTree(max_depth=2)
    model.fit(X_train, y_train)

    def test_predict(self):
        y_pred = self.model.predict(self.X_test)
        np.testing.assert_array_equal(y_pred, self.y_test)

if __name__ == '__main__':
    unittest.main()