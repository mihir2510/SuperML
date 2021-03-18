from plotly import graph_objs as go
import pandas as pd
import numpy as np


def surface_3d(stats, Z,  X='Estimator', Y=['Feature Engineering Method', 'Hyperparameter Optimisation Method'],width=750, height=750):

    x_axis_data = list(pd.unique(stats['Estimator']))
    stats['concatenated'] = stats[Y].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    y_axis = list(pd.unique(stats['concatenated']))

    y_axis_data = {}
    for index, row in stats.iterrows():
        key = row['concatenated']
        if key not in y_axis_data:
            y_axis_data[key] = []
        y_axis_data[key].append(row[Z])
    
    z_axis_data = []
    for group in y_axis_data.values():
        z_axis_data.append(group)

    fig = go.Figure(data=[go.Surface(z=z_axis_data,y=y_axis,x=x_axis_data)])
    
    fig.update_layout(title='3-D Surface Plot', autosize=True,width=width, height=height)
    
    #fig.show()

    with open('index.html', 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='https://cdnjs.cloudflare.com/ajax/libs/plotly.js/1.33.1/plotly.min.js'))