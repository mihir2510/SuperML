import plot_3d
import pandas as pd

myFile = pd.read_excel('../excel_file.xlsx', index_col=[0])
# print(myFile.columns)
plot_3d.surface_3d(myFile, 'rmse')