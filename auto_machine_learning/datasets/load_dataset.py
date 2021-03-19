import pandas as pd
import os

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
        path = os.path.dirname(__file__)
        dataset = pd.read_csv(os.path.join(path, name)+'.csv')
    except Exception as e:
        raise type(e)("Error at load_dataset. Please check the name of the dataaset")

    #Return Dataset and Label
    return dataset, label_map[name]
