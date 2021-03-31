import plot_2d,plot_3d
import pandas as pd

myfile=pd.read_excel('../excel_file.xlsx')
#plot_2d.bar_2dsubplot(myfile,Y='accuracy',plots=['Estimator','Feature Engineering Method','Hyperparameter Optimisation Method'])
#plot_2d.bar_2d(myfile, Y='accuracy', X='Estimator', groups=['Feature Engineering Method','Hyperparameter Optimisation Method'])
plot_3d.surface_3d(myfile,Z='accuracy',X='Estimator',Y=['Feature Engineering Method','Hyperparameter Optimisation Method'])