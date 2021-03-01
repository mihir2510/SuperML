import pandas as pd
import numpy as np
from data_preprocessing.preprocessing import *

class Boosting:
  def __init__(self, dataset, label, N, learner):
    self.dataset = dataset
    self.test_dataset = dataset
    self.features = features
    self.label = label
    self.N = N
    self.alphas = None
    self.models = None
    self.accuracy = []
    self.predictions = None
    self.LearnerModel = learner
  
  def fit(self):
    features = get_features(dataset, label)
    # x,y = self.dataset[self.features], self.dataset[self.label].where(self.dataset[self.label]==1,-1)
    x,y = oversample(x.copy(), y.copy(), self.features, self.label)
    evaluation = pd.DataFrame(y.copy())
    evaluation['weights'] =  1/len(self.dataset)
    # print(x.isnull().any())
    # print(y.isnull().any())
    # print(evaluation['weights'].isnull().any())

    # print(np.array(evaluation['weights']))
    alphas = []
    models = []

    for _ in range(self.N):
      print('_=',_)
      bayesSearchModel = BayesSearchCV(self.LearnerModel(), hyperparameters[self.LearnerModel])
      bayesSearchModel.fit(x, y)
      print('best params',bayesSearchModel.best_params_)
      model = self.LearnerModel(**(bayesSearchModel.best_params_))
      # print('model',model)
      # print('eval weights', evaluation['weights'])
      # print('fit before')
      model = model.fit(x, y, sample_weight=np.array(evaluation['weights']))
      # print('fit after')
      models.append(model)
      
      predictions = model.predict(x)
      score = model.score(x,y)

      evaluation['predictions'] = predictions
      evaluation['evaluation'] = np.where(evaluation['predictions'] == evaluation[self.label],1,0)
      evaluation['misclassified'] = np.where(evaluation['predictions'] != evaluation[self.label],1,0)

      accuracy = sum(evaluation['evaluation'])/len(evaluation['evaluation'])
      misclassification = sum(evaluation['misclassified'])/len(evaluation['misclassified'])
      err = np.sum(evaluation['weights']*evaluation['misclassified'])/np.sum(evaluation['weights'])

      alpha = np.log((1-err)/err)
      alphas.append(alpha)

      evaluation['weights'] *= np.exp(alpha*evaluation['misclassified'])
    
    self.alphas = alphas
    self.models = models
  
  def predict(self):
    # x_test = self.test_dataset[self.features].copy()
    # y_test = self.test_dataset[self.label].copy().where(self.dataset[self.label]==1,-1)
    x_test, y_test = self.x_test, self.y_test
    accuracy = []
    predictions = []

    for alpha,model in zip(self.alphas,self.models):
      prediction = np.round(alpha*model.predict(x_test))
      predictions.append(prediction)
      # print(np.sum(np.sum(np.array(predictions),axis=0)==y_test.values))
      # print(np.sum(np.sum(np.array(predictions),axis=0)==y_test.values)/len(predictions[0]))
      self.accuracy.append(np.sum(np.sign(np.sum(np.array(predictions),axis=0))==y_test.values)/len(predictions[0]))
    
    self.predictions = np.sum(np.array(predictions),axis=0)
    return self.predictions