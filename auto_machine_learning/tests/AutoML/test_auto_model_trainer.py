from auto_machine_learning.AutoML.auto_model_trainer import *
from auto_machine_learning.utils import check
from auto_machine_learning.datasets.load_dataset import load_dataset
from auto_machine_learning.data_preprocessing.preprocessing import dataset_split
import warnings

warnings.filterwarnings('ignore')
boston_dataset, boston_label = load_dataset('carprice')
diabetes_dataset, diabetes_label = load_dataset('diabetes')

# check function will return False because bayesian_gp doesn't work
def test_auto_trainer():
    for model in ['LinearRegression']:
        # auto_trainer(boston_dataset, boston_label, models=[model] ,task='prediction', anova_estimator=model)
        assert check(auto_trainer, boston_dataset, boston_label, models=[model], task='prediction', anova_estimator=model)
    for model in ['LogisticRegression']:
        assert check(auto_trainer, diabetes_dataset, diabetes_label, models=[model], task='classification', anova_estimator=model)
