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


from data_preprocessing.preprocessing import *
from feature_engineering.pca import featureEngineering_PCA
import pandas as pd
from feature_engineering.correlation import correlation
from feature_engineering.anova import anova_regressor
from feature_engineering.select_from_model import select_from_model
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
#from feature_engineering.feature_engineering import correlation


# dataframe = pd.read_csv('http://54.196.8.61:3000/uploads/titanic/Titanic.csv')
# label='Survived'

dataframe = pd.read_csv('http://54.196.8.61:3000/uploads/carPrice/CarPrice_Assignment.csv')
label='price'

#preprocessing testing
print(len(dataframe.columns))
print(dataframe.head())

dataframe = remove_null(dataframe, label)
# print(len(dataframe))
dataframe = label_encode(dataframe, label)
# print(dataframe.head())
# modi_data,label=oversampling(modi_data,"Survived")
# print(len(modi_data))
# print(dataframe.head())

modified_dataframe = select_from_model(dataframe, label, Lasso)
print(len(modified_dataframe.columns))
print(modified_dataframe.head())


from data_preprocessing.preprocessing import *
from feature_engineering.pca import featureEngineering_PCA
import pandas as pd

label = 'class'
dataset = pd.read_csv('http://54.196.8.61:3000/uploads/heartdataset/shorter_train.csv')
reduced_df = featureEngineering_PCA(dataset, label)[0]
print(len(dataset.columns), len(reduced_df.columns))


label = 'price'
dataset = pd.read_csv('http://54.196.8.61:3000/uploads/carPrice/CarPrice_Assignment.csv')
dataset = label_encode(dataset, label)
reduced_df = featureEngineering_PCA(dataset, label)[0]
print(len(dataset.columns), len(reduced_df.columns))

# test dataset with b = 2*a and c=3*a and d=a*a
label = 'd'
from random import randint
sz = 10
dictionary = {'a' : [randint(1,12) for i in range(sz)]}
dictionary['b'] = [i*2 for i in dictionary['a']]
dictionary['c'] = [i*3 for i in dictionary['a']] 
dictionary['d'] = [i*i for i in dictionary['a']] 
dataset = pd.DataFrame(dictionary)
print(dictionary)
reduced_df = featureEngineering_PCA(dataset, label)[0]
print(len(dataset.columns), len(reduced_df.columns))