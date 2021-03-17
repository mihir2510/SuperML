import pandas as pd
import os
# from os import path

label_map = {
    'boston': 'Target',
    'carprice': 'price',
    'diabetes': 'Outcome',
    # 'energy': '',
    'heart_disease': 'class',
    'salary': 'Salary',
    'titanic': 'Survived'
    #'weather': 'Apparent Temperature (C)'
}

def load_dataset(name):
    '''
    Returns the dataset and the label for the dataset

            Parameters:
                    name(string) : Name of the dataset

            Returns:
                    dataset(dataframe) : data for training
                    label(string) : name of the target column for the given dataset
    '''
    try:
        dataset = pd.read_csv('.\datasets\{}.csv'.format(name))
    except Exception as e:
        raise type(e)("Error at load_dataset. Please check the name of the dataaset")
    return dataset, label_map[name]
