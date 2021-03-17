from auto_machine_learning.data_preprocessing.preprocessing import *
from auto_machine_learning.feature_engineering.pca import *
import pandas as pd
from auto_machine_learning.feature_engineering.correlation import correlation
from auto_machine_learning.feature_engineering.anova import anova_regressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
#from auto_machine_learning.feature_engineering.feature_engineering import correlation
import warnings

warnings.filterwarnings('ignore')
from auto_machine_learning.data_preprocessing.preprocessing import *
from auto_machine_learning.feature_engineering.pca import *
import pandas as pd
from auto_machine_learning.feature_engineering.correlation import correlation
from auto_machine_learning.feature_engineering.anova import anova_regressor
from auto_machine_learning.feature_engineering.select_from_model import select_from_model
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier

dataframe = pd.read_csv('http://54.196.8.61:3000/uploads/titanic/Titanic.csv')
label='Survived'

#preprocessing testing
print(len(dataframe.columns))
print(dataframe.head())

dataframe = remove_null(dataframe, label)
# print(len(dataframe))
dataframe = label_encode(dataframe, label)
# print(dataframe.head())
# modi_data,label=oversampling(modi_data,"Survived")

dataframe = oversampling(dataframe, label)

correlation_matrix(dataframe,label)

# print(len(dataframe.columns))
# print(dataframe.head())
# print('label',label)
'''
modified_dataframe = select_from_model(dataframe, label, LogisticRegression)
print(len(modified_dataframe.columns))
print(modified_dataframe.head())

from auto_machine_learning.hyperparameter_optimization import hpo
trained_model = hpo_parent.get_trained_model(modified_dataframe, label, 'RandomForestClassifier', 'bayesian-tpe', 'classification',500)
X_train, X_test, Y_train, Y_test = train_test_split(modified_dataframe[get_features(modified_dataframe, label)], modified_dataframe[label], test_size=0.3, random_state = 42069)
Y_pred = trained_model.predict(X_test)
print('formula 1 score:', f1_score(Y_test, Y_pred))
print('Model Trained')
'''