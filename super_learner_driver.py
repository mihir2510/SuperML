
from auto_machine_learning.utils import get_features
from auto_machine_learning.data_preprocessing.preprocessing import *
from auto_machine_learning.feature_engineering.correlation import correlation
from auto_machine_learning.feature_engineering.anova import *
from auto_machine_learning.ensembling.super_learner import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# dataframe = pd.read_csv('http://54.196.8.61:3000/uploads/boston/Boston.csv')
# label = 'Target'
# dataframe = preprocess_data(dataframe,label, 'prediction')
# dataframe = anova_regressor(dataframe, label)
# X_train, X_test, Y_train, Y_test = dataset_split(dataframe,label)
# super_learner = SuperLearnerRegressor(['LinearRegression', 'Lasso'], 'Lasso', n_splits=2, optimize=True)
# super_learner.fit(X_train, Y_train)
# Y_pred = super_learner.predict(X_test)
# print(r2_score(Y_test, Y_pred))

dataframe = pd.read_csv('http://54.196.8.61:3000/uploads/heartdataset/shorter_train.csv')
label = 'class'
dataframe = preprocess_data(dataframe,label, 'classification')
dataframe = anova_classifier(dataframe, label)

X_train, X_test, Y_train, Y_test = dataset_split(dataframe,label)
super_learner = SuperLearnerClassifier(['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier'], 'RandomForestClassifier')
super_learner.fit(X_train, Y_train)
Y_pred = super_learner.predict(X_test)
print(f1_score(Y_test, Y_pred))

'''
datasets = {
    'Weather': {
        'filename': 'http://54.196.8.61:3000/uploads/weatherdataset/weatherHistory.csv',
        'label': 'Apparent Temperature (C)'
    },

    'Car Price': {
        'filename': 'http://54.196.8.61:3000/uploads/carPrice/CarPrice_Assignment.csv',
        'label': 'price'
    },

    'Boston': {
        'filename': 'http://54.196.8.61:3000/uploads/boston/Boston.csv',
        'label': 'Target'
    },
    
    'Diabetes': {
        'filename': 'http://54.196.8.61:3000/uploads/diabetes/pima-indians-diabetes.csv',
        'label': 'label'
    },

    'Titanic': {
        'filename': 'http://54.196.8.61:3000/uploads/titanic/Titanic.csv',
        'label': 'Survived'
    },

    'Heart_Disease': {
        'filename': 'http://54.196.8.61:3000/uploads/heartdataset/shorter_train.csv',
        'label': 'class'
    },

    'Nasa': {
        'filename': 'http://54.196.8.61:3000/uploads/nasa/nasa.csv',
        'label': 'Hazardous'
    },

    'Rain-Australia': {
        'filename': 'http://54.196.8.61:3000/uploads/weather/weatherAUS.csv',
        'label': 'RainTomorrow'
    },

    'Fitbit': {
        'filename': 'http://54.196.8.61:3000/uploads/fitbit/fitbit_dataset.csv',
        'label': 'calorie'
    }
}
'''