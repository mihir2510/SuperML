from utils import *
import numpy as np

def correlation(dataset,label,threshold=0.90):
    '''
    Uses correlation between features to drop highly correlated feature and return the processed data

            Parameters:
                    dataset(dataframe) : data to be used for training model
                    label (string): target column of the dataframe  
                    threshold(float) : threshold for deciding the features to be dropped

            Returns:
                    dataset(dataframe) : processed data to be used for training model
    '''
    #print(threshold)
    features=get_features(dataset,label)

    correlation = dataset[features].corr().abs()
    try:
        upperTriangular = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(np.bool))
    except Exception as e:
        raise type(e)("Error while creating upper triangle matrix")
    toDrop = [feature for feature in upperTriangular.columns if any(upperTriangular[feature] > threshold)]
    features = [feature for feature in features if feature not in toDrop]
    features.append(label)
    return dataset[features]
        