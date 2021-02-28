from sklearn import metrics
from math import sqrt


def get_model_metrics(model,task, X_test, Y_test):
    stats = {}
    Y_pred = model.predict(X_test)
    if task=='classifiction':
        stats['accuracy'] = metrics.accuracy_score(Y_test, Y_pred)
        stats['precision'] = metrics.precision_score(Y_test, Y_pred)
        stats['recall'] = metrics.recall_score(Y_test, Y_pred)
        stats['f1'] = metrics.f1_score(Y_test, Y_pred)
    else:
        stats['r2'] = metrics.r2_score(Y_test, Y_pred)
        stats['rmse'] = sqrt(metrics.mean_squared_error(Y_test, Y_pred))
        stats['mae'] = metrics.mean_absolute_error(Y_test, Y_pred)
    return stats