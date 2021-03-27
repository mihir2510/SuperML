from plotly import graph_objs as go
import pandas as pd
import numpy as np


def surface_3d(stats, Z,  X='Estimator', Y=['Feature Engineering Method', 'Hyperparameter Optimisation Method'],file_name='index.html',width=1500, height=1500):

    x_axis_data = list(pd.unique(stats[X]))
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
    
    fig.update_layout(title='3-D Surface Plot', autosize=False,width=width, height=height)

    fig.update_layout(scene = dict(
                    xaxis = dict(
                         backgroundcolor="rgb(200, 200, 230)",
                         gridcolor="white",
                         showbackground=True,
                         zerolinecolor="white",
                         title='.<br><br><br><br><br><br><br><br><br>'+str(X),
                         ticks='outside',
                         ticklen=60,
                         tickcolor='white',),
                    yaxis = dict(
                        backgroundcolor="rgb(230, 200,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
                        title='.<br><br><br><br><br><br><br><br><br>'+ ' +'.join(Y),
                        ticks='outside',
                        ticklen=60,
                        tickcolor='white',),
                    zaxis = dict(
                        backgroundcolor="rgb(230, 230,200)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="white",
                        title='.<br><br><br><br><br><br>'+Z),))

    #fig.update_scenes(xaxis_ticklen=4)
    #fig.show()
    fig.write_html(file_name)
'''
    with open('index.html', 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='https://cdnjs.cloudflare.com/ajax/libs/plotly.js/1.33.1/plotly.min.js'))
'''