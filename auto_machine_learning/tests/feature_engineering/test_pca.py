from auto_machine_learning.feature_engineering.pca import *
from auto_machine_learning.utils import check
from auto_machine_learning.datasets.load_dataset import load_dataset

boston_dataset, boston_label = load_dataset('boston')
diabetes_dataset, diabetes_label = load_dataset('diabetes')

def test_pca():
    assert check(pca, boston_dataset, boston_label)
    assert check(pca, diabetes_dataset, diabetes_label)
