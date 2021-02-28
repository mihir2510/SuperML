from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from constants import models

def get_features(dataset,label):
    #extracting features from dataset
    features = list(dataset.columns)
    features.remove(label)
    return features


map_model = {model.__name__: model for model in models}

def get_model(name):
    return map_model.get(name)

