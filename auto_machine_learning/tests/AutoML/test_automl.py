from auto_machine_learning.AutoML.automl import *
from auto_machine_learning.utils import check
from auto_machine_learning.datasets.load_dataset import load_dataset
from auto_machine_learning.data_preprocessing.preprocessing import dataset_split
import warnings

warnings.filterwarnings('ignore')
boston_dataset, boston_label = load_dataset('boston')
diabetes_dataset, diabetes_label = load_dataset('diabetes')

def test_automl():
    for anova_estimator in ['LinearRegression', 'RandomForestRegressor']:
        assert check(auto_ensemble, boston_dataset, boston_label, task='prediction',base_layer_models=['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor','GradientBoostingRegressor'],meta_layer_model=anova_estimator, max_evals=10)
    for anova_estimator in ['LogisticRegression', 'RandomForestClassifier']:
        assert check(auto_ensemble, diabetes_dataset, diabetes_label, task='classification',base_layer_models=['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier','ExtraTreesClassifier','AdaBoostClassifier'] ,meta_layer_model=anova_estimator, max_evals=10)

def test_auto_ensemble_trainer():
    for anova_estimator in ['LinearRegression', 'RandomForestRegressor']:
        assert check(automl_run, boston_dataset, boston_label, task='prediction',base_layer_models=['LinearRegression', 'Ridge', 'Lasso', 'DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor','GradientBoostingRegressor'],meta_layer_models=[anova_estimator], max_evals=10)
    for anova_estimator in ['LogisticRegression', 'RandomForestClassifier']:
        assert check(automl_run, diabetes_dataset, diabetes_label, task='classification',base_layer_models=['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier','ExtraTreesClassifier','AdaBoostClassifier'] ,meta_layer_models=[anova_estimator], max_evals=10)
