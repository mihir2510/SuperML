from data_preprocessing.preprocessing import *
from datasets import load_dataset
from utils import check

def test_preprocess_data():
    dataset, label = load_dataset('boston')
    check(preprocess_data,dataset, label, 'regression')

