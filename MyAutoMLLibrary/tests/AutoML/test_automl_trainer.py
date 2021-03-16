from MyAutoMLLibrary.AutoML.auto_trainer import *
from MyAutoMLLibrary.utils import check
from MyAutoMLLibrary.datasets.load_dataset import load_dataset
from MyAutoMLLibrary.data_preprocessing.preprocessing import dataset_split
import warnings

warnings.filterwarnings('ignore')
boston_dataset, boston_label = load_dataset('boston')
diabetes_dataset, diabetes_label = load_dataset('diabetes')

def test_auto_trainer():
    for meta_model in ['LinearRegression', 'RandomForestRegressor']:
        check(auto_trainer, boston_dataset, boston_label, task='prediction', meta_model=meta_model)
    for meta_model in ['LogisticRegression', 'RandomForestClassifier']:
        check(auto_trainer, diabetes_dataset, diabetes_label, task='classification', meta_model=meta_model)
