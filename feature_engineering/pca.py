from sklearn.decomposition import PCA
from data_preprocessing.preprocessing import get_features

import pandas as pd


# The number of features in PCA is estimated with the help of Minka's 
# MLE - Most Likelihood Estimation
def featureEngineering_PCA(dataset, features):
    X = dataset[features]
    pca = PCA(n_components='mle', svd_solver='auto')
    X = pca.fit_transform(X)
    return [pd.DataFrame(X), []]

if __name__ == '__main__':
    dataset = pd.read_csv('http://54.196.8.61:3000/uploads/heartdataset/shorter_train.csv')
    features = get_features(dataset, 'class')
    reduced_df = featureEngineering_PCA(dataset, features)[0]
    print(reduced_df.head())
