from utils import *
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn import metrics
from math import ceil, sqrt, log2

def anova_regressor(dataset,label,modelClass=RandomForestRegressor):
    features = get_features(dataset, label)
    n = len(features)
    # List containing the different values to consider as K
    numberOfFeatures = [ceil(log2(n)), ceil(sqrt(n)), n]
    X,Y = dataset[features], dataset[label]
    optimal_k = -1
    max_score = float('-inf')
    for k in numberOfFeatures:
        selector = SelectKBest(f_classif,k=k)
        selector.fit(X,Y)
        columns=selector.get_support(indices=True)
        important_features = X.iloc[:,columns].columns

        X_reduced=dataset[important_features]
        X_train,X_test,Y_train,Y_test = train_test_split(X_reduced,Y,test_size=0.3,random_state=1)
        model=modelClass()
        model.fit(X_train,Y_train)

        score=model.score(X_test,Y_test)

        if score>max_score:
            max_score = score
            optimal_k = k
        # print('internal',k,score, max_score)
        # print('internal',optimal_k)

    selector = SelectKBest(f_classif,k=optimal_k)
    selector.fit(X,Y)
    column  = selector.get_support(indices=True)
    important_features = list(X.iloc[:,columns].columns)
    # print(important_features)
    important_features.append(label)
    X = dataset[important_features]
    return X

def anova_classifier(dataset,label,modelClass=RandomForestClassifier):
    features = get_features(dataset, label)
    n = len(features)
    # List containing the different values to consider as K
    numberOfFeatures = [ceil(log2(n)), ceil(sqrt(n)), n]
    X,Y = dataset[features], dataset[label]
    optimal_k = -1
    max_score = float('-inf')
    for k in numberOfFeatures:
        selector = SelectKBest(f_classif,k=k)
        selector.fit(X,Y)
        columns=selector.get_support(indices=True)
        important_features = X.iloc[:,columns].columns

        X_reduced=dataset[important_features]
        X_train,X_test,Y_train,Y_test = train_test_split(X_reduced,Y,test_size=0.3,random_state=1)
        model=modelClass()
        model.fit(X_train,Y_train)

        Y_pred = model.predict(X_test)
        score=metrics.f1_score(Y_test,Y_pred)

        if score>max_score:
            max_score = score
            optimal_k = k
    
    selector = SelectKBest(f_classif,k=optimal_k)
    selector.fit(X,Y)
    column  = selector.get_support(indices=True)
    important_features = list(X.iloc[:,columns].columns)
    important_features.append(label)
    X = dataset[important_features]
    return X