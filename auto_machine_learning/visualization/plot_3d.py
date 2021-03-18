from plotly import graph_objs as go
import pandas as pd
import numpy as np


def surface_3d(stats, Z,  X='Estimator', Y=['Feature Engineering Method', 'Hyperparameter Optimisation Method']):

    #stats['concatenated'] = stats['Feature Engineering Method'] + ' ' + stats['Hyperparameter Optimisation Method']
    x = list(pd.unique(stats['Estimator']))
    stats['concatenated'] = stats[Y].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    y = list(pd.unique(stats['concatenated']))

    xy = {}
    for index, row in stats.iterrows():
        key = row['concatenated']
        if key not in xy:
            xy[key] = []
        xy[key].append(row[Z])
    
    print(x)
    print(y)

    z = []
    for group in xy.values():
        z.append(group)

    fig = go.Figure(data=[go.Surface(z=z,y=y,x=x)])
    
    fig.update_layout(title='', autosize=True,width=1000, height=1000)
    
    #fig.show()

    with open('index.html', 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='https://cdnjs.cloudflare.com/ajax/libs/plotly.js/1.33.1/plotly.min.js'))