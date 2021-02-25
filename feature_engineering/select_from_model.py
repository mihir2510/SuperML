from data_preprocessing.preprocessing import get_features
from sklearn.feature_selection import SelectFromModel

def select_from_model(dataframe, label, model_class):
    selector = SelectFromModel(estimator=model_class())
    