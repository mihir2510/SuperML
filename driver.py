from data_preprocessing.preprocessing import *
from feature_engineering.pca import featureEngineering_PCA
import pandas as pd
from feature_engineering.correlation import correlation
from feature_engineering.anova import anova_regressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier

#from feature_engineering.feature_engineering import correlation
import warnings

warnings.filterwarnings('ignore')
from data_preprocessing.preprocessing import *
from feature_engineering.pca import featureEngineering_PCA
import pandas as pd
from feature_engineering.correlation import correlation
from feature_engineering.anova import anova_regressor
from feature_engineering.select_from_model import select_from_model
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier

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

from hyperparameter_optimization import hpo_parent
trained_model = hpo_parent.get_trained_model(modified_dataframe, label, 'LinearRegression', 'bayesian-tpe', 'prediction')
print('Model Trained')