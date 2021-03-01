from utils import get_features
from data_preprocessing.preprocessing import *
from feature_engineering.correlation import correlation
from feature_engineering.anova import *
from ensembling.super_learner import *
from AutoML.automl import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# dataframe = pd.read_csv('http://54.196.8.61:3000/uploads/heartdataset/shorter_train.csv')
# label = 'class'

# ensemble = automl(dataframe, label, 'classification')

dataframe = pd.read_csv('http://54.196.8.61:3000/uploads/boston/Boston.csv')
label = 'Target'

ensemble = automl(dataframe, label, 'prediction')