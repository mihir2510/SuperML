from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from utils import get_features
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def preprocess_data(dataset,label, task='classification'):
    dataset = remove_null(dataset,label)
    dataset = label_encode(dataset,label) 
    if task == 'classification':
        dataset = oversampling(dataset,label)
    #correlation_matrix(dataset,label)
    return dataset
    # return oversampling(label_encode(remove_null(dataset,label),label),label)

def remove_null(dataset,label):

    #get features
    features = get_features(dataset,label)
    
    # Removing Columns with more than 50% null data
    for feature in features:
        if (dataset[feature].isnull().sum()/len(dataset) > 0.5):
            dataset.drop([feature], axis=1, inplace=True)
            features.remove(feature)

    # Removing rows having null values
    dataset.dropna(inplace=True)

    if 'Unnamed: 0' in dataset.columns:
        dataset.drop(['Unnamed: 0'],axis=1,inplace=True)

    return dataset
    
def label_encode(dataset,label):
    #get features
    features = get_features(dataset,label)
    
    #for features
    for feature in features:
        if dataset[feature].dtype == object:
            encoder = LabelEncoder()
            encoder.fit(dataset[feature])
            dataset[feature] = encoder.transform(dataset[feature])
    #for label
    if dataset[label].dtype == object:
        dataset[label] = LabelEncoder().fit_transform(dataset[label])
        
    if 'Unnamed: 0' in dataset.columns:
        dataset.drop(['Unnamed: 0'],axis=1,inplace=True)
    return dataset

def oversampling(dataset, label):
    #get features
    features = get_features(dataset,label)
    new_dataset = None
    if True:
        oversampler=RandomOverSampler()
        X,Y=oversampler.fit_resample(dataset[features],dataset[label])

        new_dataset = pd.DataFrame(X,columns=features)
        new_dataset[label] = pd.DataFrame(Y)
    else:
        new_dataset = dataset
    if 'Unnamed: 0' in new_dataset.columns:
        new_dataset.drop(['Unnamed: 0'],axis=1,inplace=True)
    
    return new_dataset

def dataset_split(dataset,label, test_size=0.3, random_state = 1):
    features = get_features(dataset, label)
    X, Y = dataset[features], dataset[label]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, Y_train, Y_test


def correlation_matrix(dataset,label):
    features = get_features(dataset,label)
    correlation = dataset[features].corr().abs()
    f, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(correlation, cmap='coolwarm', annot=True, ax=ax)
    plt.show()
    

