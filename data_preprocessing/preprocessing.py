from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

def get_features(dataset,label):
    
    #extracting features from dataset
    features = list(dataset.columns)
    features.remove(label)
    return features

def remove_null(dataset,label):

    #get features
    features=get_features(dataset,label)
    
    # Removing Columns with more than 50% null data
    for feature in features:
        if (dataset[feature].isnull().sum()/len(dataset) > 0.5):
            dataset.drop([feature], axis=1, inplace=True)
            features.remove(feature)

    # Removing rows having null values
    dataset.dropna(inplace=True)
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
    return dataset

def oversampling(dataset, label):
    #get features
    features = get_features(dataset,label)

    if len(dataset) <=1500:
        oversampler=RandomOverSampler()
        X,Y=oversampler.fit_resample(dataset[features],dataset[label])

        X=pd.DataFrame(X,columns=features)
        Y=pd.DataFrame(Y,columns=dataset[label])

    if 'Unnamed: 0' in X.columns:
        X.drop(['Unnamed: 0'],axis=1,inplace=True)

    return X,Y