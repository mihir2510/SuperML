from auto_machine_learning.metrics.metrics import *
from auto_machine_learning.utils import check
from auto_machine_learning.datasets.load_dataset import load_dataset
from auto_machine_learning.data_preprocessing.preprocessing import dataset_split
from sklearn.linear_model import LinearRegression, LogisticRegression
import warnings

warnings.filterwarnings('ignore')
boston_dataset, boston_label = load_dataset('boston')
diabetes_dataset, diabetes_label = load_dataset('diabetes')

def test_get_model_metrics():
    boston_train_X, boston_test_X, boston_train_y, boston_test_y = dataset_split(boston_dataset, boston_label)
    model = LinearRegression()
    model.fit(boston_train_X, boston_train_y)
    assert check(get_model_metrics, model, [], 'prediction', boston_test_X, boston_test_y)
    diabetes_train_X, diabetes_test_X, diabetes_train_y, diabetes_test_y = dataset_split(diabetes_dataset, diabetes_label)
    model = LogisticRegression()
    model.fit(diabetes_train_X, diabetes_train_y)
    assert check(get_model_metrics, model, [0,1], 'classification', diabetes_test_X, diabetes_test_y)

def test_get_model_metrics_ensemble():
    boston_train_X, boston_test_X, boston_train_y, boston_test_y = dataset_split(boston_dataset, boston_label)
    model = LinearRegression()
    model.fit(boston_train_X, boston_train_y)
    assert check(get_model_metrics_ensemble, [], 'prediction', boston_test_y, model.predict(boston_test_X))
    diabetes_train_X, diabetes_test_X, diabetes_train_y, diabetes_test_y = dataset_split(diabetes_dataset, diabetes_label)
    model = LogisticRegression()
    model.fit(diabetes_train_X, diabetes_train_y)
    assert check(get_model_metrics_ensemble, [0,1], 'classification', diabetes_test_y, model.predict(diabetes_test_X))
