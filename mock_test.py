from hyper_parameter_optimization.test_hpo import HPO
import pandas as pd
dataset = pd.read_csv('http://54.196.8.61:3000/uploads/heartdataset/shorter_train.csv')
model = HPO(dataset,'class','RandomForestClassifier')

testX = [[0, 49.207124, 0,
4.0, 162.99616699999999, 181.10868200000002,
0, 0, 148.227858, 1, 0.9445469999999999,
2, 0, 3]]

print(len(testX[0])) # output: 14 
print(model.predict(testX)) # output: 1