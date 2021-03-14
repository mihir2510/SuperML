import plotly.graph_objects as go
import pandas as pd




def bar_2d(stats):

    print(stats)

    stats['concatenated'] = stats['Feature Engineering Method'] + ' ' + stats['Hyperparameter Optimisation Method']
    xaxis_data=pd.unique(stats['Estimator'])
    xy={}
    for index, row in stats.iterrows():
        key = 'concatenated'
        if row[key] not in xy:
            xy[row[key]] = []
        xy[row[key]].append(row['r2'])
    print(xy)

    bar=[]
    for group in xy:
        bar.append(go.Bar(name=group, x=xaxis_data, y=xy[group]))

    fig = go.Figure(data=bar)
    
    # Change the bar mode
    fig.update_layout(barmode='group')

    fig.update_yaxes(range=(stats['r2'].min()-0.05, stats['r2'].max()+0.05))
    
    #fig.show()

    with open('index.html', 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='https://cdnjs.cloudflare.com/ajax/libs/plotly.js/1.33.1/plotly.min.js'))