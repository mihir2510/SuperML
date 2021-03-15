from MyAutoMLLibrary.feature_engineering.correlation import *
from MyAutoMLLibrary.utils import check
from MyAutoMLLibrary.datasets.load_dataset import load_dataset

boston_dataset, boston_label = load_dataset('boston')
titanic_dataset, titanic_label = load_dataset('titanic')

def test_correlation():
    check(correlation, boston_dataset, boston_label)
    check(correlation, titanic_dataset, titanic_label)
