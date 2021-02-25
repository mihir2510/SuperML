from data_preprocessing.preprocessing import get_features
import numpy as np

def correlation(dataset,label,threshold=0.90):
    print(threshold)
    features=get_features(dataset,label)

    correlation = dataset[features].corr().abs()

    upperTriangular = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(np.bool))
    toDrop = [feature for feature in upperTriangular.columns if any(upperTriangular[feature] > threshold)]
    features = [feature for feature in features if feature not in toDrop]
    return dataset[features]
        