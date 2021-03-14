from plotly import graph_objs as go
import pandas as pd
import numpy as np


def surface_3d(stats):

    stats['concatenated'] = stats['Feature Engineering Method'] + ' ' + stats['Hyperparameter Optimisation Method']
    x = list(pd.unique(stats['Estimator']))
    y = list(pd.unique(stats['concatenated']))

    xy = {}
    for index, row in stats.iterrows():
        key = row['concatenated']
        if key not in xy:
            xy[key] = []
        xy[key].append(row['r2'])
    
    print(x)
    print(y)

    z = []
    for group in xy.values():
        z.append(group)

    # x = ['Linear','Ridge','Lasso']
    # y = ['PCA','ANOVA','SKLEARN']
    # z = [[0.85, 0.92, 0.93],[0.88, 0.95, 0.96],[0.84, 0.915, 0.94]]

    fig = go.Figure(data=[go.Surface(z=z,y=y,x=x)])
    
    fig.update_layout(title='', autosize=True,width=1000, height=1000)
    
    fig.show()
    
    with open('index.html', 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='https://cdnjs.cloudflare.com/ajax/libs/plotly.js/1.33.1/plotly.min.js'))