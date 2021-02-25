from data_preprocessing.preprocessing import *
from feature_engineering.pca import featureEngineering_PCA
import pandas as pd
from feature_engineering.correlation import correlation
from feature_engineering.anova import anova_regressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor, AdaBoostClassifier, ExtraTreesClassifier, ExtraTreesRegressor, BaggingClassifier, BaggingRegressor, GradientBoostingClassifier, GradientBoostingRegressor
#from feature_engineering.feature_engineering import correlation
import warnings

warnings.filterwarnings('ignore')
'''
dataset = pd.read_csv('http://54.196.8.61:3000/uploads/titanic/Titanic.csv')
label='Survived'

#preprocessing testing
modi_data = remove_null(dataset,label)
print(len(label))
modi_data = label_encode(modi_data,label)
print(modi_data.head()) # this is modi data, by KD
'''

label = 'd'
from random import randint
sz = 100
dictionary = {'a' : [randint(1,sz) for i in range(sz)]}
dictionary['b'] = [i*2 for i in dictionary['a']]
dictionary['c'] = [i*3 for i in dictionary['a']] 
dictionary['d'] = [i*i for i in dictionary['a']] 
dataset = pd.DataFrame(dictionary)

reduced_df = anova_regressor(dataset, label,LinearRegression)
print(len(dataset.columns), len(reduced_df.columns))

# dataset = pd.read_csv('http://54.196.8.61:3000/uploads/heartdataset/shorter_train.csv')
# reduced_df = featureEngineering_PCA(dataset, 'class')[0]
# print(len(dataset.columns), len(reduced_df.columns))


# dataset = pd.read_csv('http://54.196.8.61:3000/uploads/carPrice/CarPrice_Assignment.csv')
# features = get_features(dataset, 'price')
# reduced_df = featureEngineering_PCA(dataset, features)[0]
# print(len(dataset.columns), len(reduced_df.columns))
