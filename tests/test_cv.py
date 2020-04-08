import unittest
from src.cv import get_ith_fold, k_fold_cross_validation
import pandas as pd
from sklearn.naive_bayes import GaussianNB


class TestCrossValidation(unittest.TestCase):

    def test_get_ith_fold(self):
        df = pd.DataFrame({'a': [i for i in range(0, 11)],
                           'b': [2*i for i in range(0, 11)]})
        indices = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

        train, test = get_ith_fold(5, df, 3)

        self.assertListEqual(list(train.index),
                             list(range(0, 6)) + [8, 9, 10])
        self.assertListEqual(list(test.index), [6, 7])

        train, test = get_ith_fold(5, df, 4)

        self.assertListEqual(list(train.index),
                             list(range(0, 8)))
        self.assertListEqual(list(test.index), [8, 9, 10])

    def test_k_fold(self):
        model = GaussianNB()
        X = [[i, 2*i] for i in range(0, 10)]
        y = [[0], [0], [0], [0], [0],
             [1], [1], [1], [1], [1]]
        k_fold_cross_validation(5, X, y, model)