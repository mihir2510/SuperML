from auto_machine_learning.feature_engineering.anova import *
from auto_machine_learning.utils import check
from auto_machine_learning.datasets.load_dataset import load_dataset

boston_dataset, boston_label = load_dataset('boston')
diabetes_dataset, diabetes_label = load_dataset('diabetes')

def test_anova_regressor():
    assert check(anova_regressor, boston_dataset, boston_label)
    assert check(anova_regressor, boston_dataset, boston_label, 'LinearRegression')
    # assert check(anova_regressor, diabetes_dataset, diabetes_label) == False

def test_anova_classifier():
    assert check(anova_classifier, diabetes_dataset, diabetes_label)
    assert check(anova_classifier, diabetes_dataset, diabetes_label, 'LogisticRegression')
    # assert check(anova_classifier, boston_dataset, boston_label) == False
