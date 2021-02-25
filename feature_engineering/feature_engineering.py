
def get_features(dataset,label):
    
    #extracting features from dataset
    features = list(dataset.columns)
    features.remove(label)
    return features

def correlation(dataset,label,threshold=0.90):

    correlation = dataset[features].corr().abs()
    '''f, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(correlation, cmap='coolwarm', annot=True, ax=ax)'''

    upperTriangular = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(np.bool))
    toDrop = [feature for feature in upperTriangular.columns if any(upperTriangular[feature] > 0.95)]
    features = [feature for feature in features if feature not in toDrop]
    return [dataset[features], features]

        