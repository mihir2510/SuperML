from MyAutoMLLibrary.ensembling.super_learner import *
from MyAutoMLLibrary.utils import check
from MyAutoMLLibrary.datasets.load_dataset import load_dataset
from MyAutoMLLibrary.data_preprocessing.preprocessing import dataset_split
import warnings

warnings.filterwarnings('ignore')
boston_dataset, boston_label = load_dataset('boston')
diabetes_dataset, diabetes_label = load_dataset('diabetes')

def test_SuperLearnerRegressor():
    def test_func(meta_model, new_models):
        meta_model = meta_model or 'RandomForestRegressor'
        super_learner_obj = SuperLearnerRegressor(meta_model=meta_model)
        if new_models:
            super_learner_obj.add_models(new_models)
        boston_train_X, boston_test_X, boston_train_y, boston_test_y = dataset_split(boston_dataset, boston_label)
        super_learner_obj.fit(boston_train_X, boston_train_y)
        super_learner_obj.predict(boston_test_X)
    new_models = ['DecisionTreeRegressor', 'RandomForestRegressor', 'ExtraTreesRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor']
    # assert check(test_func, None, None)
    # assert check(test_func, 'AdaBoostRegressor', None)
    assert check(test_func, 'AdaBoostRegressor', new_models)
    assert check(test_func, None, new_models)
    assert check(test_func, 'LinearRegression', None)
    assert check(test_func, 'LinearRegression', new_models)


def test_SuperLearnerClassifier():
    def test_func(meta_model, new_models):
        meta_model = meta_model or 'RandomForestClassifier'
        super_learner_obj = SuperLearnerClassifier(meta_model=meta_model)
        if new_models:
            super_learner_obj.add_models(new_models)
        diabetes_train_X, diabetes_test_X, diabetes_train_y, diabetes_test_y = dataset_split(diabetes_dataset, diabetes_label)
        super_learner_obj.fit(diabetes_train_X, diabetes_train_y)
        super_learner_obj.predict(diabetes_test_X)
    new_models = ['DecisionTreeClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier']
    assert check(test_func, 'AdaBoostClassifier', None)
    assert check(test_func, 'AdaBoostClassifier', new_models)
    assert check(test_func, None, new_models)
    assert check(test_func, 'LogisticRegression', None)
    assert check(test_func, 'LogisticRegression', new_models)
