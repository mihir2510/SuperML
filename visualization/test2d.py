import plot_2d
import pandas as pd

# x=['Linear', 'Ridge', 'Lasso']
# group=['PCA','ANOVA']
# y=[[0.85, 0.92, 1],[0.83, 0.99, 8]]



# xy={
#     'PCA':[0.85, 0.92, 1],
#     'ANOVA' :[0.83, 0.99, 8]
# }

myFile = pd.read_excel('../excel_file.xlsx', index_col=[0])
stats_list = myFile
plot_2d.bar_2d(stats_list)