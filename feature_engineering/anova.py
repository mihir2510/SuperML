from data_preprocessing.preprocessing import get_features
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.model_selection import train_test_split

def anova_feature(dataset,label,modelClass):
    features = get_features(dataset)
    n = len(features)
    # List containing the different values to consider as K
    numberOfFeatures = [ceil(log2(n)), ceil(sqrt(n)), n]
    X,Y = dataset[features], dataset[label]
    optimal_k = -1
    max_score = float('-inf')
    for k in numberOfFeatures:
        selector = SelectKBest(f_classif,k=k)
        selector = selector.get_support(indices=True)
        columns=selector.get_support(indices=True)
        important_features = X.iloc[:,column].columns

        X_reduced=dataset[important_features]
        X_train,X_test,Y_train,Y_test=train_test_split(X_reduced,test_size=0.3,random_state=1)
        model=modelClass()
        


    model = modelClass()
    model.fit(XTrain, YTrain)

    if isClassification:
      YPred = model.predict(XTest)
      score = metrics.f1_score(YTest, YPred)
    else:
      score = model.score(XTest, YTest)
    
    if score > max_score:
      max_score = score
      optimal_k = k
  
  if 0 >= optimal_k or optimal_k > n:
    k = n

  selector = SelectKBest(f_classif, k=optimal_k)
  selector.fit(X, Y)
  columns = selector.get_support(indices=True)
  importantFeatures = X.iloc[:, columns].columns
  X = dataset[importantFeatures]

  return [X, importantFeatures]