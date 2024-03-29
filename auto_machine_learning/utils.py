from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pickle
import pandas as pd

#---------------------------------------------------------------------------------------------------------------------#

models = [LinearRegression, Ridge, Lasso, DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor, LogisticRegression, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, DecisionTreeClassifier]

#---------------------------------------------------------------------------------------------------------------------#

def get_features(dataset,label):
    '''
    Returns the list of strings containing the names of features in the data set

            Parameters:
                    dataset (dataframe) : contains data for model training
                    label (string) : the column name which is the target of machine learning 

            Returns:
                    features (string)
    '''
    try:
        features = list(dataset.columns)
        features.remove(label)
    except Exception as e:
        raise type(e)("Check the label name")
    return features


map_model = {model.__name__: model for model in models}

#---------------------------------------------------------------------------------------------------------------------#

def get_model(name):
    '''
    Returns the model corresponding to the model name

            Parameters:
                    name (string) : name of the model required

            Returns:
                    model (model class reference) 
    '''
    try:
        return map_model.get(name)
    except Exception as e:
        raise type(e)("Check the model name")

#---------------------------------------------------------------------------------------------------------------------#

def pickle_model(model,file_name='pickled_model'):
    '''
    Saves the model in a pickle format for later use

            Parameters:
                    model (model object) 


           
    '''
    
    pickle.dump(model, open(file_name+'.sav', 'wb'))
    print('\n'+'Model Downloaded')
#---------------------------------------------------------------------------------------------------------------------#

def get_csv(pd_stats,filename='excel_file'):
    pd_stats.to_excel(filename+'.xlsx')
    print('\n'+'Excel File Generated')

#---------------------------------------------------------------------------------------------------------------------#

def download_dataset(dataset_path):
    try:
        return pd.read_csv(dataset_path)
    except Exception as e:
        raise type(e)(" Check the dataset_path ")

#---------------------------------------------------------------------------------------------------------------------#

def check(func, *args, **kw):
    try:
        func(*args, **kw)
        return True
    except Exception:
        return False

#---------------------------------------------------------------------------------------------------------------------#

