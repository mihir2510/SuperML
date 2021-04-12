from auto_machine_learning.utils import *
from sklearn.feature_selection import SelectFromModel
import pandas as pd

#---------------------------------------------------------------------------------------------------------------------#

def select_from_model(dataset, label, select_from_model_class):
    '''
    Implements the select from model method

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe  
                    select_from_model_class(class of the model): model which will be used for deciding 

            Returns:
                    dataset(dataframe) : processed data to be used for training model
    '''

    try:
        if select_from_model_class.__name__ =='BaggingRegressor':
            select_from_model_class=map_model['RandomForestRegressor']
        elif select_from_model_class.__name__ =='BaggingClassifier':
            select_from_model_class=map_model['RandomForestClassifier'] 
        selector = SelectFromModel(estimator=select_from_model_class())
        X, Y = dataset[get_features(dataset, label)], dataset[[label]]
        selector.fit(X, Y)
        columns = selector.get_support(indices=True)
    except Exception as e:
        raise type(e)("Error in select_from_model")

    important_columns = list(X.iloc[:, columns].columns)
    important_columns.append(label)

    modified_dataset = dataset[important_columns]

    
    return modified_dataset

#---------------------------------------------------------------------------------------------------------------------#