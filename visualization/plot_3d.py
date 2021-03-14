from plotly import graph_objs as go
import pandas as pd
import numpy as np


def surface_3d(stats_list):
    # print(stats_list)

    x = list(set([row[0] for row in stats_list]))
    y = list(set([row[1] for row in stats_list]))

    print(x)
    print(y)
    # x = ['Linear','Ridge','Lasso']
    # y = ['PCA','ANOVA','SKLEARN']
    # z = [[0.85, 0.92, 0.93],[0.88, 0.95, 0.96],[0.84, 0.915, 0.94]]

    # fig = go.Figure(data=[go.Surface(z=z,y=y,x=x)])

    # fig.update_layout(title='', autosize=True,
    #                 width=500, height=500)
                    
    # fig.show()