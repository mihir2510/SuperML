from sklearn.model_selection import train_test_split
from utils import get_model, get_features
from hyperparameter_optimization.hpo_methods import *


def get_trained_model(dataset, label, model_name, task, method_name='standard', max_evals=100, test_size=0.3, random_state=1):
    '''
    Train the model with given data and hpo method . Returns the trained model

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe  
                    task (string) : type of task 
                    model_name(string) : name of the model on which data is to be trained
                    method_name(string) : Name of the hyper parameter method to be used while training
                    max_evals(int) : Number of evaluators
                    test_size(float) : Fraction of the data to be used for testing
                    random_state(int) : Random state to be used

            Returns:
                    model (model object) : the trained model on which hpo is performed
    '''
    features = get_features(dataset, label)
    X, Y = dataset[features], dataset[label]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state = random_state)
    model = get_model(model_name)
    if method_name=='standard':
        model=model()
        trained_model = model.fit(X_train,Y_train)
    elif method_name == 'grid_search':
        trained_model = grid_search(model, X_train, Y_train)
    elif method_name == 'random_search':
        trained_model = random_search(model, X_train, Y_train)
    elif method_name == 'bayesian_gp':
        trained_model = bayesian_gp(model, X_train, Y_train)
    elif method_name == 'bayesian_tpe':
        trained_model = bayesian_tpe(model, X_train, X_test, Y_train, Y_test, task, max_evals=max_evals)
    else:
        raise Exception("'No hpo method named {}'.format(method_name)")
    return trained_model

'''
def generate_stats(dataset, label, model_names, method_names, task, max_evals=100, test_size=0.3, random_state = 1):
    
    features = get_features(dataset, label)
    X, Y = dataset[features], dataset[label]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state = random_state)
    function_map =  {
        'grid': grid_search,
        'random': random_search,
        'bayesian-gp': bayesian_gp,
        'bayesian-tpe': bayesian_tpe
    }
    stats = []
    for model_name in model_names:
        for method in method_names:
            # Do something
            model = get_trained_model(dataset, label, model_name, method, task, max_evals, test_size, random_state)
            score = model.score(X_test, Y_test)
            # TODO: add all metrics for classfication as well
            stats.append((model_name, method, score))
    return stats
    # return sorted(stats, key=lambda stat : stat[2], reverse=True) # sort on basis of score/accuracy
'''