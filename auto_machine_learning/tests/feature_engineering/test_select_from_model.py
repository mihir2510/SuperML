from auto_machine_learning.feature_engineering.select_from_model import *
from auto_machine_learning.utils import check
from sklearn.linear_model import LinearRegression, LogisticRegression
from auto_machine_learning.datasets.load_dataset import load_dataset

boston_dataset, boston_label = load_dataset('boston')
diabetes_dataset, diabetes_label = load_dataset('diabetes')

def test_select_from_model():
    assert check(select_from_model, boston_dataset, boston_label, LinearRegression)
    assert check(select_from_model, diabetes_dataset, diabetes_label, LogisticRegression)
