import plot_2d,plot_3d
import pandas as pd

myfile=pd.read_excel('../excel_file.xlsx')
plot_2d.bar_2dsubplot(myfile,Y='accuracy')
#plot_3d.surface_3d(myfile,Z='accuracy')