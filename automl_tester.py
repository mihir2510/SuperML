from auto_machine_learning.utils import get_features
from auto_machine_learning.data_preprocessing.preprocessing import *
from auto_machine_learning.feature_engineering.correlation import correlation
from auto_machine_learning.feature_engineering.anova import *
from auto_machine_learning.ensembling.super_learner import *
from auto_machine_learning.AutoML import *
from auto_machine_learning.AutoML.auto_ensemble_trainer import *
import warnings
warnings.filterwarnings('ignore')

dataframe = pd.read_csv('http://54.196.8.61:3000/uploads/heartdataset/shorter_train.csv')
label = 'class'

pd_stats, ensemble = auto_ensemble_trainer(dataframe, label, 'classification', ['LogisticRegression', 'DecisionTreeClassifier'], ['RandomForestClassifier'])

# dataframe = pd.read_csv('http://54.196.8.61:3000/uploads/boston/Boston.csv')
# label = 'Target'

# pd_stats, ensemble = auto_ensemble_trainer(dataframe, label, 'prediction', ['LinearRegression', 'Lasso', 'Ridge'], ['LinearRegression', 'Lasso', 'Ridge'])

print(pd_stats)
