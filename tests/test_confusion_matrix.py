import unittest
from src.confusion_matrix import ConfusionMatrix
import pandas as pd
from sklearn.naive_bayes import GaussianNB


class TestConfusionMatrix(unittest.TestCase):

    def test_metrics(self):
        cnf = ConfusionMatrix(10, 20, 40, 30)

        assert cnf.get_accuracy() == 40/100
        assert cnf.get_precision() == 10/30
        assert cnf.get_recall() == 10/50
        assert cnf.get_error() == 60/100
        assert cnf.get_sensitivity() == cnf.get_recall()
        assert cnf.get_specificity() == 30/50
