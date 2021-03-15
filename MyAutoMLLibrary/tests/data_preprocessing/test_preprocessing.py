from MyAutoMLLibrary.data_preprocessing.preprocessing import *
from MyAutoMLLibrary.utils import check
from MyAutoMLLibrary.datasets.load_dataset import load_dataset

boston_dataset, boston_label = load_dataset('boston')
titanic_dataset, titanic_label = load_dataset('titanic')

def test_preprocess_data():
    assert check(preprocess_data, boston_dataset, boston_label, 'regression')
    assert check(preprocess_data, boston_dataset, boston_label, 'classification') == False

def test_remove_null():
    assert check(remove_null, boston_dataset, boston_label)
    assert check(remove_null, titanic_dataset, titanic_label)

def test_label_encode():
    assert check(label_encode, titanic_dataset, titanic_label)

def test_oversampling():
    assert check(oversampling, boston_dataset, boston_label) == False
    assert check(oversampling, titanic_dataset, titanic_label)

def test_dataset_split():
    assert check(dataset_split, boston_dataset, boston_label, test_size=0.3)
    split_datasets = dataset_split(boston_dataset, boston_label, test_size=0.3)
    other_split_datasets = dataset_split(boston_dataset, boston_label, test_size=0.3)
    # the datasets should be same since the random_state is same
    assert [(split_datasets[i] == other_split_datasets[i]).all().all() for i in range(4)] == [True]*4
    # alternate_split_datasets = dataset_split(boston_dataset, boston_label, test_size=0.3, random_state = 630)
    # assert [(split_datasets[i] == alternate_split_datasets[i]).all().all() for i in range(4)] == [True]*4

def test_correlation_matrix():
    assert check(correlation_matrix, boston_dataset, boston_label)
    assert check(correlation_matrix, titanic_dataset, titanic_label)
