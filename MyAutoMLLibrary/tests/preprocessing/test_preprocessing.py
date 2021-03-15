from MyAutoMLLibrary.data_preprocessing.preprocessing import *
from MyAutoMLLibrary.utils import check
from MyAutoMLLibrary.datasets.load_dataset import load_dataset

def test_preprocess_data():
    # assert 1==1
    dataset, label = load_dataset('boston')
    assert check(preprocess_data, dataset, label, 'regression')
    assert check(preprocess_data, dataset, label, 'classification') == False

