from MyAutoMLLibrary.hyperparameter_optimization.hpo import *
from MyAutoMLLibrary.data_preprocessing.preprocessing import dataset_split
from MyAutoMLLibrary.utils import check
from MyAutoMLLibrary.datasets.load_dataset import load_dataset
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')
boston_dataset, boston_label = load_dataset('boston')
diabetes_dataset, diabetes_label = load_dataset('diabetes')

def test_hpo():
    for model in [LinearRegression, Lasso, Ridge]:
        for method_name in ['grid_search', 'random_search', 'bayesian_tpe']:
            check(get_trained_model, boston_dataset, boston_label, model.__name__, 'prediction', method_name)
    for model in [LogisticRegression, RandomForestClassifier]:
        for method_name in ['grid_search', 'random_search', 'bayesian_tpe']:
            check(get_trained_model, heart_dataset, heart_label, model.__name__, 'prediction', method_name)
