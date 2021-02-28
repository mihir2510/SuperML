from AutoML.result_generator import *
from utils import map_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
from feature_engineering.anova import anova_regressor
import warnings
from data_preprocessing.preprocessing import *

warnings.filterwarnings('ignore')
models = list(map_model.keys())
# models = models[9:]
# print(models)
# models = ['LogisticRegression', 'RandomForestClassifier']
models = ['LinearRegression', 'Lasso', 'Ridge']
dataset = pd.read_csv('http://54.196.8.61:3000/uploads/boston/Boston.csv')
print('dataset downloaded')
label = 'Target'
# dataset.drop(['Unnamed: 0'],axis=1,inplace=True)
# dataset = anova_regressor(dataset, label, LinearRegression)
# X_train, X_test, Y_train, Y_test = dataset_split(dataset, label)
# print('here',X_train.shape)
# params = {'fit_intercept': True, 'n_jobs': 1, 'normalize': True}
# base_model = LinearRegression(**params)
# base_model.fit(X_train, Y_train)
# print('Accuracy for base model:',base_model.score(dataset[get_features(dataset,label)], dataset[label]))

stats = generate_results(dataset, label, 'prediction', models=models, modelClass=RandomForestRegressor, hpo_methods=['standard','grid_search','random_search', 'bayesian_tpe'],sortby='r2')#])

"""with open('stats2.txt','w') as f:
f.write(str(stats))"""

print(stats)


