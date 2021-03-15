from MyAutoMLLibrary.feature_engineering.anova import *
from MyAutoMLLibrary.utils import check
from MyAutoMLLibrary.datasets.load_dataset import load_dataset

boston_dataset, boston_label = load_dataset('boston')
titanic_dataset, titanic_label = load_dataset('titanic')

def test_anova_regressor():
    check(anova_regressor,boston_dataset, boston_label)
    check(anova_regressor,boston_dataset, boston_label, 'LinearRegression')
    check(anova_regressor,titanic_dataset, titanic_label) == False

def test_anova_classifier():
    check(anova_classifier, titanic_dataset, titanic_label)
    check(anova_classifier, titanic_dataset, titanic_label, 'LogisticRegression')
    check(anova_classifier,boston_dataset, boston_label) == False
