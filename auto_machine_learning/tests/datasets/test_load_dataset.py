from auto_machine_learning.utils import check
from auto_machine_learning.datasets.load_dataset import load_dataset

def test_load_dataset():
    for name in ['boston', 'carprice', 'diabetes', 'heart_disease', 'salary', 'titanic']:
        assert check(load_dataset, name)
    
    assert check(load_dataset, 'literally_any_other_name') == False
# test_load_dataset()