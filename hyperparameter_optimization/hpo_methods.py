from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, f1_score
from skopt import BayesSearchCV
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from utils import get_features
from constants import *

def grid_search(model, X_train, Y_train):
    grid_search_model = GridSearchCV(model(), hyperparameters[model])
    grid_search_model.fit(X_train, Y_train)
    optimized_model = model(**grid_search_model.best_params_)
    optimized_model.fit(X_train, Y_train)
    return optimized_model

def random_search(model, X_train, Y_train):
    random_search_model = RandomizedSearchCV(model(), hyperparameters[model])
    random_search_model.fit(X_train, Y_train)

    optimized_model = model(**random_search_model.best_params_)
    optimized_model.fit(X_train, Y_train)
    return optimized_model

def bayesian_gp(model, X_train, Y_train):
    bayesian_gp_model = BayesSearchCV(model(), hyperparameters[model])
    bayesian_gp_model.fit(X_train, Y_train)

    optimized_model = model(**bayesian_gp_model.best_params_)
    optimized_model.fit(X_train, Y_train)
    return optimized_model 
    
def bayesian_tpe(model, X_train, X_test, Y_train, Y_test, task, max_evals=100):
    def objective_func(space):
        hyperopt_model = model(**space)
        hyperopt_model.fit(X_train, Y_train)
        Y_pred = hyperopt_model.predict(X_test)

        loss = -r2_score(Y_test, Y_pred) if task == 'prediction' else -f1_score(Y_test, Y_pred)
        return {'loss': loss, 'status': STATUS_OK}
    
    best_hyperparameters = fmin(fn=objective_func, space=hyperopt_hyperparameters[model], algo=tpe.suggest, trials=Trials(), max_evals=max_evals)
    
    best_hyperparameters_values = {}
    for hyperparameter, index in best_hyperparameters.items():
        best_hyperparameters_values[hyperparameter] = hyperparameters[model][hyperparameter][index]
    
    optimized_model = model(**best_hyperparameters_values)
    optimized_model.fit(X_train, Y_train)

    return optimized_model
