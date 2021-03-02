from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from hyperopt import hp

hyperparameters = {
    LinearRegression: {
        'fit_intercept': [True, False],
        'normalize': [True, False],
        'n_jobs': [1]
    },

    Ridge: {
        'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
        'fit_intercept': [True, False],
        'normalize': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    },

    Lasso: {
        'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
        'fit_intercept': [True, False],
        'normalize': [True, False],
        'precompute': [True, False],
        'warm_start': [True, False],
        'selection': ['cyclic', 'random']
    },

    DecisionTreeRegressor: {
        'criterion': ['mse', 'friedman_mse', 'mae'],
        'splitter': ['best', 'random'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'ccp_alpha': [0, 0.01, 0.1, 1]
    },

    RandomForestRegressor: {
        'n_estimators': [20, 40, 60, 80, 100],
        'criterion': ['mse', 'mae'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'ccp_alpha': [0, 0.01, 0.1, 1]
    },

    ExtraTreesRegressor: {
        'n_estimators': [20, 40, 60, 80, 100],
        'criterion': ['mse', 'mae',],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'ccp_alpha': [0, 0.01, 0.1, 1]
    },

    GradientBoostingRegressor: {
        'n_estimators' : [20,40,60,80,100],
        'criterion' : ['friedman_mse', 'mse', 'mae'],
        'max_features' : ['auto', 'sqrt', 'log2'],
        'ccp_alpha': [0, 0.01, 0.1, 1]
    },

    AdaBoostRegressor: {
        'base_estimator' : [RandomForestRegressor,DecisionTreeRegressor],
        'n_estimators' : [25,50,75,100]
    },

    LogisticRegression: {
        'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
        'fit_intercept': [True, False],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [1000],
        'warm_start': [True, False]
    },

    DecisionTreeClassifier: {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'class_weight': ['balanced', None],
        'ccp_alpha': [0, 0.01, 0.1, 1]
    },

    RandomForestClassifier: {
        'n_estimators': [20, 40, 60, 80, 100],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'ccp_alpha': [0, 0.01, 0.1, 1]
    },

    ExtraTreesClassifier: {
        'n_estimators': [20, 40, 60, 80, 100],
        'criterion': ['gini', 'entropy'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'ccp_alpha': [0, 0.01, 0.1, 1]
    },
    GradientBoostingClassifier: {
        'n_estimators' : [20, 40, 60, 80, 100],
        'criterion' : ['friedman_mse', 'mse', 'mae'],
        'max_features' : ['auto', 'sqrt', 'log2'],
        'ccp_alpha': [0, 0.01, 0.1, 1]
    },

    AdaBoostClassifier: {
        'base_estimator' : [RandomForestClassifier,DecisionTreeClassifier],
        'n_estimators' : [25,50,75,100]
    }

}



hyperopt_hyperparameters = {}

for model in hyperparameters:
    hyperopt_hyperparameters[model] = {}
    for hyperparameter in hyperparameters[model]:
        hyperopt_hyperparameters[model][hyperparameter] = hp.choice(hyperparameter, hyperparameters[model][hyperparameter])

