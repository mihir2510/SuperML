from MyAutoMLLibrary.feature_engineering.pca import *
from MyAutoMLLibrary.utils import check
from MyAutoMLLibrary.datasets.load_dataset import load_dataset

boston_dataset, boston_label = load_dataset('boston')
titanic_dataset, titanic_label = load_dataset('titanic')

def test_pca():
    check(pca, boston_dataset, boston_label)
    check(pca, titanic_dataset, titanic_label)
