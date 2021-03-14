import plotly.graph_objects as go




def bar_2d(xaxis_data,xy):
    
    
    
    bar=[]
    for xy.values in xy:
        bar.append(go.Bar(name=_group_data, x=xaxis_data, y=[0.85, 0.92, 1]))


    # fig = go.Figure(data=[
    #     go.Bar(name='PCA', x=xaxis_data, y=[0.85, 0.92, 1]),
    #     go.Bar(name='ANOVA', x=xaxis_data, y=[0.88, 0.95, 1])
    # ])
    fig = go.Figure(data=bar)
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.update_yaxes(range=(0.8, 1))
    fig.show()