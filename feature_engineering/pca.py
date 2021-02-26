from sklearn.decomposition import PCA
from utils import *
import pandas as pd


# The number of features in PCA is estimated with the help of Minka's 
# MLE - Most Likelihood Estimation
def pca(dataset, label):
    features = get_features(dataset, label)
    X = dataset[features]
    pca = PCA(n_components='mle', svd_solver='auto')
    X = pca.fit_transform(X)
    X = pd.DataFrame(X)
    X[label] = dataset[label]
    return X
    # return [pd.DataFrame(X), []]

