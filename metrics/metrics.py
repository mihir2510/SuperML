from sklearn import metrics
from math import sqrt


def get_model_metrics(model,task, X_test, Y_test):
    '''
    Returns the dictionary cotaining metrics for the given data.

            Parameters:
                    model (trained ml model) 
                    task (string): ml task prediction or classification
                    X test (dataframe): test data
                    Y test (dataframe): test labels 

            Returns:
                    stats (dictionary): contains the metrics for given data
    '''
    stats = {}
    Y_pred = model.predict(X_test)
    if task=='classification':
        stats['accuracy'] = metrics.accuracy_score(Y_test, Y_pred)
        stats['precision_micro'] = metrics.precision_score(Y_test, Y_pred, average='micro')
        stats['precision_macro'] = metrics.precision_score(Y_test, Y_pred, average='macro')
        stats['recall'] = metrics.recall_score(Y_test, Y_pred)
        stats['f1'] = metrics.f1_score(Y_test, Y_pred)
    else:
        stats['r2'] = metrics.r2_score(Y_test, Y_pred)
        stats['rmse'] = sqrt(metrics.mean_squared_error(Y_test, Y_pred))
        stats['mae'] = metrics.mean_absolute_error(Y_test, Y_pred)
    return stats

def get_model_metrics_ensemble(task, Y_test, Y_pred):
    '''
    Returns the dictionary cotaining metrics for the given data.

            Parameters:
                    task (string): ml task prediction or classification
                    Y_test (dataframe): true labels
                    Y_pred (dataframe): predicted labels


            Returns:
                    stats (dictionary): contains the metrics for given data
    '''
    stats = {}
    if task=='classification':
        stats['accuracy'] = metrics.accuracy_score(Y_test, Y_pred)
        stats['precision_micro'] = metrics.precision_score(Y_test, Y_pred, average='micro')
        stats['precision_macro'] = metrics.precision_score(Y_test, Y_pred, average='macro')
        stats['recall'] = metrics.recall_score(Y_test, Y_pred)
        stats['f1'] = metrics.f1_score(Y_test, Y_pred)
    else:
        stats['r2'] = metrics.r2_score(Y_test, Y_pred)
        stats['rmse'] = -sqrt(metrics.mean_squared_error(Y_test, Y_pred))
        stats['mae'] = -metrics.mean_absolute_error(Y_test, Y_pred)
    return stats