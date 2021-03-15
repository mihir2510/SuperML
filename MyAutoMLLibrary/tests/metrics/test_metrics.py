from MyAutoMLLibrary.metrics.metrics import *
from MyAutoMLLibrary.utils import check
from MyAutoMLLibrary.datasets.load_dataset import load_dataset
from MyAutoMLLibrary.data_preprocessing.preprocessing import dataset_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings

warnings.filterwarnings('ignore')
boston_dataset, boston_label = load_dataset('boston')
diabetes_dataset, diabetes_label = load_dataset('diabetes')

def test_get_model_metrics():
    boston_train_X, boston_test_X, boston_train_y, boston_test_y = dataset_split(boston_dataset, boston_label)
    model = LinearRegression()
    model.fit(boston_train_X, boston_train_y)
    check(get_model_metrics, model, [], 'prediction', boston_test_X, boston_test_y)
    diabetes_train_X, diabetes_test_X, diabetes_train_y, diabetes_test_y = dataset_split(diabetes_dataset, diabetes_label)
    model = LogisticRegression()
    model.fit(diabetes_train_X, diabetes_train_y)
    check(get_model_metrics, model, [], 'classification', diabetes_test_X, diabetes_test_y)

def test_get_model_metrics_ensemble():
    boston_train_X, boston_test_X, boston_train_y, boston_test_y = dataset_split(boston_dataset, boston_label)
    model = LinearRegression()
    model.fit(boston_train_X, boston_train_y)
    check(get_model_metrics_ensemble, model, [], 'prediction', boston_test_y, model.predict(boston_test_X))
    diabetes_train_X, diabetes_test_X, diabetes_train_y, diabetes_test_y = dataset_split(diabetes_dataset, diabetes_label)
    model = LogisticRegression()
    model.fit(diabetes_train_X, diabetes_train_y)
    check(get_model_metrics_ensemble, model, [], 'classification', diabetes_test_X, model.predict(diabetes_test_X))
