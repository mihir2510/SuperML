from sklearn.model_selection import train_test_split
from utils import get_model, get_features
from hyperparameter_optimization.hpo import *

def get_trained_model(dataset, label, model_name, method_name, task, max_evals=100, test_size=0.3, random_state=1):

    features = get_features(dataset, label)
    X, Y = dataset[features], dataset[label]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state = random_state)
    model = get_model(model_name)

    if method_name == 'grid':
        trained_model = grid_search(model, X_train, Y_train)
    elif method_name == 'random':
        trained_model = random_search(model, X_train, Y_train)
    elif method_name == 'bayesian-gp':
        trained_model = bayesian_gp(model, X_train, Y_train)
    elif method_name == 'bayesian-tpe':
        trained_model = bayesian_tpe(model, X_train, X_test, Y_train, Y_test, task, max_evals=max_evals)
    else:
        print('No hpo method named {}'.format(method_name))
    return trained_model

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
