from sklearn import metrics
from math import sqrt
import pandas as pd


def get_model_metrics(model,label_data,task, X_test, Y_test):
    '''
    Returns the dictionary cotaining metrics for the given data.

            Parameters:
                    model (trained ml model)
                    label_data : to check number of classes  
                    task (string): ml task prediction or classification
                    X test (dataframe): test data
                    Y test (dataframe): test labels 
            Returns:
                    stats (dictionary): contains the metrics for given data
    '''

    number_of_classes = len(pd.unique(label_data))
    stats = {}
    Y_pred = model.predict(X_test)
    if task=='classification':
        stats['accuracy'] = metrics.accuracy_score(Y_test, Y_pred)
        if number_of_classes==2:
            stats['precision'] = metrics.precision_score(Y_test, Y_pred)
            stats['recall'] = metrics.recall_score(Y_test, Y_pred)
            stats['f1'] = metrics.f1_score(Y_test, Y_pred)
        else:
            stats['precision_micro'] = metrics.precision_score(Y_test, Y_pred, average='micro')
            stats['precision_macro'] = metrics.precision_score(Y_test, Y_pred, average='macro')
            stats['recall_micro'] = metrics.recall_score(Y_test, Y_pred,average='micro')
            stats['recall_macro'] = metrics.recall_score(Y_test, Y_pred,average='macro')
            stats['f1_micro'] = metrics.f1_score(Y_test, Y_pred,average='micro')
            stats['f1_macro'] = metrics.f1_score(Y_test, Y_pred,average='macro')
    else:
        stats['r2'] = metrics.r2_score(Y_test, Y_pred)
        stats['rmse'] = sqrt(metrics.mean_squared_error(Y_test, Y_pred))
        stats['mae'] = metrics.mean_absolute_error(Y_test, Y_pred)
    return stats


def get_model_metrics_ensemble(label_data,task, Y_test, Y_pred):

    '''
    Returns the dictionary cotaining metrics for the given data.

            Parameters:
                    label_data : to check number of classes
                    task (string): ml task prediction or classification
                    Y_test (dataframe): true labels
                    Y_pred (dataframe): predicted labels
            Returns:
                    stats (dictionary): contains the metrics for given data
    '''

    number_of_classes = len(pd.unique(label_data))
    stats = {}
    if task=='classification':
        stats['accuracy'] = metrics.accuracy_score(Y_test, Y_pred)
        if number_of_classes==2:
            stats['precision'] = metrics.precision_score(Y_test, Y_pred)
            stats['recall'] = metrics.recall_score(Y_test, Y_pred)
            stats['f1'] = metrics.f1_score(Y_test, Y_pred)
        else:
            stats['precision_micro'] = metrics.precision_score(Y_test, Y_pred, average='micro')
            stats['precision_macro'] = metrics.precision_score(Y_test, Y_pred, average='macro')
            stats['recall_micro'] = metrics.recall_score(Y_test, Y_pred,average='micro')
            stats['recall_macro'] = metrics.recall_score(Y_test, Y_pred,average='macro')
            stats['f1_micro'] = metrics.f1_score(Y_test, Y_pred,average='micro')
            stats['f1_macro'] = metrics.f1_score(Y_test, Y_pred,average='macro')
    else:
        stats['r2'] = metrics.r2_score(Y_test, Y_pred)
        stats['rmse'] = -sqrt(metrics.mean_squared_error(Y_test, Y_pred))
        stats['mae'] = -metrics.mean_absolute_error(Y_test, Y_pred)
    return stats