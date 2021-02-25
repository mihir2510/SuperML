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