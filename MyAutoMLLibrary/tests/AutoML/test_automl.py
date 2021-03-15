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
        check(automl, boston_dataset, boston_label, task='prediction',base_layer_models=['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor','GradientBoostingRegressor'],meta_layer_model=modelClass)
    for modelClass in ['LogisticRegression', 'RandomForestClassifier']:
        check(automl, diabetes_dataset, diabetes_label, task='classification',base_layer_models=['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier','ExtraTreesClassifier','AdaBoostClassifier'] ,meta_layer_model=modelClass)
