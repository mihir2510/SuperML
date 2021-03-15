import pandas as pd
# from os import path

label_map = {
    'boston': 'Target',
    'carprice': 'price',
    'diabetes': 'label',
    # 'energy': '',
    'hear_disease': 'class',
    'salary': 'Salary',
    'titanic': 'Survived',
    'weather': 'Apparent Temperature (C)'
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
    dataset = pd.read_csv('./datasets/{}.csv'.format(name))
    return dataset, label_map[name]