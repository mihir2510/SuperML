import plotly.graph_objects as go
import pandas as pd

def bar_2d(stats, Y, X='Estimator', groups=['Feature Engineering Method','Hyperparameter Optimisation Method']):
    
    x_axis_data = list(pd.unique(stats[X]))
    stats['concatenated'] = stats[groups].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    y_axis_data={}

    for index, row in stats.iterrows():
        key = 'concatenated'
        if row[key] not in y_axis_data:
            y_axis_data[row[key]] = []
        y_axis_data[row[key]].append(row[Y])

    bar=[]
    for group in y_axis_data:
        bar.append(go.Bar(name=group, x=x_axis_data, y=y_axis_data[group]))

    fig = go.Figure(data=bar)
    
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.update_yaxes(range=(stats[Y].min()-0.05, stats[Y].max()+0.05))
    
    #fig.show()

    with open('index.html', 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='https://cdnjs.cloudflare.com/ajax/libs/plotly.js/1.33.1/plotly.min.js'))