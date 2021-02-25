from data_preprocessing.preprocessing import removenull,labelencode,oversampling
import pandas as pd
#from feature_engineering.feature_engineering import correlation


boston = pd.read_csv('http://54.196.8.61:3000/uploads/titanic/Titanic.csv')

'''
#preprocessing testing
print(boston.head())
print(len(boston))
modi_data = removenull(boston,"Survived")
print(len(boston))
modi_data = labelencode(modi_data,"Survived")
print(modi_data.head())
modi_data,label=oversampling(modi_data,"Survived")
print(len(modi_data))
'''
#print(correlation(1,2))