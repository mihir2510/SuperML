from auto_machine_learning.hyperparameter_optimization.hpo_methods import *
from auto_machine_learning.data_preprocessing.preprocessing import dataset_split
from auto_machine_learning.utils import check
from auto_machine_learning.datasets.load_dataset import load_dataset
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')
boston_dataset, boston_label = load_dataset('boston')
diabetes_dataset, diabetes_label = load_dataset('diabetes')
boston_X_train, boston_X_test, boston_y_train, boston_y_test = dataset_split(boston_dataset, boston_label)
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = dataset_split(diabetes_dataset, diabetes_label)

def test_grid_search():
    for model in [LinearRegression, Lasso, Ridge]:
        # grid_search(boston_dataset, boston_label, model)
        assert check(grid_search, model, boston_X_train, boston_y_train)
    for model in [LogisticRegression, RandomForestClassifier]:
        assert check(grid_search, model, diabetes_X_train, diabetes_y_train)

def test_random_search():
    for model in [LinearRegression, Lasso, Ridge]:
        assert check(random_search, model, boston_X_train, boston_y_train)
    for model in [LogisticRegression, RandomForestClassifier]:
        assert check(random_search, model, diabetes_X_train, diabetes_y_train)

def test_bayesian_tpe():
    for model in [LinearRegression, Lasso, Ridge]:
        assert check(bayesian_tpe, model, boston_X_train, boston_X_test, boston_y_train, boston_y_test, 'prediction', max_evals = 10)
    for model in [LogisticRegression, RandomForestClassifier]:
        assert check(bayesian_tpe, model, diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test, 'classification', max_evals = 10)
