from auto_machine_learning.hyperparameter_optimization.hpo import *
from auto_machine_learning.data_preprocessing.preprocessing import dataset_split
from auto_machine_learning.utils import check
from auto_machine_learning.datasets.load_dataset import load_dataset
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')
boston_dataset, boston_label = load_dataset('boston')
diabetes_dataset, diabetes_label = load_dataset('diabetes')

def test_hpo():
    for model in [LinearRegression, Lasso, Ridge]:
        for method_name in ['grid_search', 'random_search', 'bayesian_tpe']:
            # get_trained_model(boston_dataset, boston_label, model.__name__, 'prediction', method_name)
            assert check(get_trained_model, boston_dataset, boston_label, model.__name__, 'prediction', method_name, max_evals=10)
    for model in [LogisticRegression, RandomForestClassifier]:
        for method_name in ['grid_search', 'random_search', 'bayesian_tpe']:
            assert check(get_trained_model, diabetes_dataset, diabetes_label, model.__name__, 'classification', method_name, max_evals=10)
