from sklearn.ensemble import RandomForestClassifier
models = {
    'RandomForestClassifier': RandomForestClassifier
}

def HPO(dataset, label, string):
    model = models[string]()
    features = list(dataset.columns)
    features.remove(label)
    features.remove('Unnamed: 0')
    X = dataset[features]
    y = dataset[label]
    print(X.shape, y.shape)
    model.fit(X,y)
    return model

