from MyAutoMLLibrary.AutoML.automl import *
from MyAutoMLLibrary.utils import check
from MyAutoMLLibrary.datasets.load_dataset import load_dataset
from MyAutoMLLibrary.data_preprocessing.preprocessing import dataset_split
import warnings

warnings.filterwarnings('ignore')
boston_dataset, boston_label = load_dataset('boston')
diabetes_dataset, diabetes_label = load_dataset('diabetes')

def test_automl():
    for modelClass in ['LinearRegression', 'RandomForestRegressor']:
        check(auto_ensemble, boston_dataset, boston_label, task='prediction',base_layer_models=['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor','GradientBoostingRegressor'],meta_layer_model=modelClass)
    for modelClass in ['LogisticRegression', 'RandomForestClassifier']:
        check(auto_ensemble, diabetes_dataset, diabetes_label, task='classification',base_layer_models=['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier','ExtraTreesClassifier','AdaBoostClassifier'] ,meta_layer_model=modelClass)

def test_auto_ensemble_trainer():
    for modelClass in ['LinearRegression', 'RandomForestRegressor']:
        check(automl_run, boston_dataset, boston_label, task='prediction',base_layer_models=['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor','GradientBoostingRegressor'],modelClass=modelClass)
    for modelClass in ['LogisticRegression', 'RandomForestClassifier']:
        check(automl_run, diabetes_dataset, diabetes_label, task='classification',base_layer_models=['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier','ExtraTreesClassifier','AdaBoostClassifier'] ,modelClass=modelClass)
