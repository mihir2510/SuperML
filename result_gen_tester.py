from auto_machine_learning.automl.auto_model_trainer import *
from auto_machine_learning.automl.automl import *

from auto_machine_learning.utils import map_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
from auto_machine_learning.feature_engineering.anova import anova_regressor
import warnings
from auto_machine_learning.data_preprocessing.preprocessing import *
from auto_machine_learning.datasets.load_dataset import load_dataset
from auto_machine_learning.visualization.plot_2d import *

warnings.filterwarnings('ignore')
models = list(map_model.keys())
# models = models[9:]
# print(models)
# models = ['LogisticRegression', 'RandomForestClassifier']
#models = ['BaggingRegressor']
#dataset = pd.read_csv('http://54.196.8.61:3000/uploads/titanic/Boston.csv')
dataset,label = load_dataset('titanic')
print('dataset downloaded')
# dataset.drop(['Unnamed: 0'],axis=1,inplace=True)
# dataset = anova_regressor(dataset, label, LinearRegression)
# X_train, X_test, Y_train, Y_test = dataset_split(dataset, label)
# print('here',X_train.shape)
# params = {'fit_intercept': True, 'n_jobs': 1, 'normalize': True}
# base_model = LinearRegression(**params)
# base_model.fit(X_train, Y_train)
# print('Accuracy for base model:',base_model.score(dataset[get_features(dataset,label)], dataset[label]))

#stats,model =automl_run(dataset, label, task='classification',excel_file='1',sortby='f1')

#stats,model= auto_trainer(dataset,label,task='prediction',feature_engineering_methods= ['all_features','anova'], hpo_methods=['standard','bayesian_tpe'], models=[] ,anova_estimator=None, sortby='r2',excel_file='1')

#stats,model= auto_trainer(dataset,label,task='classification',feature_engineering_methods= ['all_features','anova'], hpo_methods=['standard','bayesian_tpe'], models=[] ,anova_estimator=None, sortby='f1',excel_file='1')


"""with open('stats2.txt','w') as f:
f.write(str(stats))"""

#print(stats)

myFile = pd.read_excel('2.xlsx')
stats_list = myFile.values.tolist()
bar_2d(myFile,Y='F1 Score',X='Model',groups=['Task'])
