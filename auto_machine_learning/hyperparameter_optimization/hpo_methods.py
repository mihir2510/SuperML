from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, f1_score
from skopt import BayesSearchCV
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from auto_machine_learning.utils import get_features
from auto_machine_learning.constants import *

def grid_search(model, X_train, Y_train):
    '''
    Traditonal grid search method. It evaluates all the possible combinations

            Parameters:
                    model(class of model) : class of the model on which the hpo is to be performed
                    X_train(dataframe): the dataframe containing data(features) for training
                    Y_train (pd_series) : labels for the training

            Returns:
                    model (model object) : the trained model on which hpo is performed
    '''
    try:
        grid_search_model = GridSearchCV(model(), hyperparameters[model])
        grid_search_model.fit(X_train, Y_train)
        optimized_model = model(**grid_search_model.best_params_)
        optimized_model.fit(X_train, Y_train)
    except Exception as e:
        raise type(e)("Error at grid_search. Check data and model")
    return optimized_model

def random_search(model, X_train, Y_train):
    '''
    Random Search for HPO. Randomly picks combinations to evaluate best hyper parameters

            Parameters:
                    model(class of model) : class of the model on which the hpo is to be performed
                    X_train(dataframe): the dataframe containing data(features) for training
                    Y_train (pd_series) : labels for the training

            Returns:
                    model (model object) : the trained model on which hpo is performed
    '''
    try:
        random_search_model = RandomizedSearchCV(model(), hyperparameters[model])
        random_search_model.fit(X_train, Y_train)

        optimized_model = model(**random_search_model.best_params_)
        optimized_model.fit(X_train, Y_train)
    except Exception as e:
        raise type(e)("Error at random_search. Check data and model")
    return optimized_model

def bayesian_gp(model, X_train, Y_train):
    '''
    Uses bayes theorem with gaussian search to continuously evaluate hyper parameters.

            Parameters:
                    model(class of model) : class of the model on which the hpo is to be performed
                    X_train(dataframe): the dataframe containing data(features) for training
                    Y_train (pd_series) : labels for the training

            Returns:
                    model (model object) : the trained model on which hpo is performed
    '''
    try:
        bayesian_gp_model = BayesSearchCV(model(), hyperparameters[model])
        bayesian_gp_model.fit(X_train, Y_train)

        optimized_model = model(**bayesian_gp_model.best_params_)
        optimized_model.fit(X_train, Y_train)
    except Exception as e:
        raise type(e)("Error at bayesian_gp. Check data and model")
    return optimized_model 
    
def bayesian_tpe(model, X_train, X_test, Y_train, Y_test, task, max_evals=100):
    '''
    Uses bayesian theorem with tree base parzen estimators.

            Parameters:
                    model(class of model) : class of the model on which the hpo is to be performed
                    X_train(dataframe): the dataframe containing data(features) for training
                    Y_train (pd_series) : labels for the training
                    X_test(dataframe): the dataframe containing data(features) for for testing
                    Y_test (pd_series) : labels for testing/evaluating the trained model
                    task (string) : type of task default is classification
                    max_evals(int) : max number of evaluations to be done


            Returns:
                    model (model object) : the trained model on which hpo is performed
    '''
    def objective_func(space):
        '''
        function to optimize the search

                Parameters:
                        space(dictionary): hyperparameter grid

                Returns:
                        dictionary : loss and status
        '''
        hyperopt_model = model(**space)
        hyperopt_model.fit(X_train, Y_train)
        Y_pred = hyperopt_model.predict(X_test)

        loss = -r2_score(Y_test, Y_pred) if task == 'prediction' else -f1_score(Y_test, Y_pred)
        return {'loss': loss, 'status': STATUS_OK}
    try:
        best_hyperparameters = fmin(fn=objective_func, space=hyperopt_hyperparameters[model], algo=tpe.suggest, trials=Trials(), max_evals=max_evals)
        
        best_hyperparameters_values = {}
        for hyperparameter, index in best_hyperparameters.items():
            best_hyperparameters_values[hyperparameter] = hyperparameters[model][hyperparameter][index]
        
        optimized_model = model(**best_hyperparameters_values)
        optimized_model.fit(X_train, Y_train)
    except Exception as e:
        raise type(e)("Error at bayesian_tpe. Check data and model")

    return optimized_model
