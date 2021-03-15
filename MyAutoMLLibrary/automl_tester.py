from utils import get_features
from data_preprocessing.preprocessing import *
from feature_engineering.correlation import correlation
from feature_engineering.anova import *
from ensembling.super_learner import *
from AutoML.automl import *
from AutoML.auto_ensemble_trainer import *
import warnings
warnings.filterwarnings('ignore')

dataframe = pd.read_csv('http://54.196.8.61:3000/uploads/heartdataset/shorter_train.csv')
label = 'class'

pd_stats, ensemble = auto_ensemble_trainer(dataframe, label, 'classification', ['LogisticRegression', 'DecisionTreeClassifier'], ['RandomForestClassifier'])

# dataframe = pd.read_csv('http://54.196.8.61:3000/uploads/boston/Boston.csv')
# label = 'Target'

# pd_stats, ensemble = auto_ensemble_trainer(dataframe, label, 'prediction', ['LinearRegression', 'Lasso', 'Ridge'], ['LinearRegression', 'Lasso', 'Ridge'])

print(pd_stats)