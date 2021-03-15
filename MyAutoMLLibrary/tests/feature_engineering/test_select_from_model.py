from MyAutoMLLibrary.feature_engineering.select_from_model import *
from MyAutoMLLibrary.utils import check
from sklearn.linear_model import LinearRegression, LogisticRegression
from MyAutoMLLibrary.datasets.load_dataset import load_dataset

boston_dataset, boston_label = load_dataset('boston')
titanic_dataset, titanic_label = load_dataset('titanic')

def test_select_from_model():
    check(select_from_model, boston_dataset, boston_label, LinearRegression)
    check(select_from_model, titanic_dataset, titanic_label, LogisticRegression)
    check(select_from_model, boston_dataset, boston_label, LogisticRegression) == False
    check(select_from_model, titanic_dataset, titanic_label, LinearRegression) == False
