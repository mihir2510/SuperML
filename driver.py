from data_preprocessing.preprocessing import *
from feature_engineering.pca import featureEngineering_PCA
import pandas as pd
from feature_engineering.correlation import correlation
#from feature_engineering.feature_engineering import correlation


boston = pd.read_csv('http://54.196.8.61:3000/uploads/titanic/Titanic.csv')
label='Survived'
'''
#preprocessing testing
print(boston.head())
print(len(boston))
modi_data = removenull(boston,"Survived")
print(len(boston))
modi_data = labelencode(modi_data,"Survived")
print(modi_data.head()) # this is modi data, by KD
modi_data,label=oversampling(modi_data,"Survived")
print(len(modi_data))
'''

modi_data=correlation(boston,label,0.8)
print(modi_data.head())

dataset = pd.read_csv('http://54.196.8.61:3000/uploads/heartdataset/shorter_train.csv')
features = get_features(dataset, 'class')
reduced_df = featureEngineering_PCA(dataset, features)[0]
print(reduced_df.head())

dataset = pd.read_csv('http://54.196.8.61:3000/uploads/nasa/nasa.csv')
features = get_features(dataset, 'Hazardous')
reduced_df = featureEngineering_PCA(dataset, features)[0]
print(reduced_df.head())