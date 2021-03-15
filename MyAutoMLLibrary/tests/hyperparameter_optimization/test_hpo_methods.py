from MyAutoMLLibrary.hyperparameter_optimization.hpo_methods import *
from MyAutoMLLibrary.data_preprocessing.preprocessing import dataset_split
from MyAutoMLLibrary.utils import check
from MyAutoMLLibrary.datasets.load_dataset import load_dataset
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

boston_dataset, boston_label = load_dataset('boston')
heart_dataset, heart_label = load_dataset('heart_disease')

def test_grid_search():
    for model in [LinearRegression, Lasso, Ridge]:
        check(grid_search, boston_dataset, boston_label, model)
    for model in [LogisticRegression, RandomForestClassifier]:
        check(grid_search, heart_dataset, heart_label, model)

def test_random_search():
    for model in [LinearRegression, Lasso, Ridge]:
        check(random_search, boston_dataset, boston_label, model)
    for model in [LogisticRegression, RandomForestClassifier]:
        check(random_search, heart_dataset, heart_label, model)

def test_bayesian_tpe():
    for model in [LinearRegression, Lasso, Ridge]:
        check(bayesian_tpe, boston_dataset, boston_label, model)
    for model in [LogisticRegression, RandomForestClassifier]:
        check(bayesian_tpe, heart_dataset, heart_label, model)
