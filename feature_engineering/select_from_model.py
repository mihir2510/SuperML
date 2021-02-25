from utils import *
from sklearn.feature_selection import SelectFromModel
import pandas as pd

def select_from_model(dataset, label, model_class):
    selector = SelectFromModel(estimator=model_class())
    X, Y = dataset[get_features(dataset, label)], dataset[label]
    selector.fit(X, Y)
    columns = selector.get_support(indices=True)
    important_columns = list(X.iloc[:, columns].columns)
    important_columns.append(label)

    modified_dataset = dataset[important_columns]
    return modified_dataset