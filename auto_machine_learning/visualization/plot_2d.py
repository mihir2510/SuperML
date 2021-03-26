import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots


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
        bar.append(go.Bar(name=group, x=x_axis_data, y=y_axis_data[group],text=y_axis_data[group],textposition='outside'))

    fig = go.Figure(data=bar)
    
    # Change the bar mode
    fig.update_layout(barmode='group',legend_title_text = "") #showlegend=False,
    fig.update_xaxes(title_text="")

    fig.update_yaxes(title_text="",range=(stats[Y].min()-0.05, stats[Y].max()+0.05))

    
    #fig.show()
    #fig.write_image("fig1.png")
    fig.write_html("index.html")

'''
    with open('index.html', 'w') as f:
        f.write(fig.to_html(full_html=True, include_plotlyjs='https://cdnjs.cloudflare.com/ajax/libs/plotly.js/1.33.1/plotly.min.js'))
'''

def bar_2dsubplot(stats, Y, plots=['Estimator','Feature Engineering Method','Hyperparameter Optimisation Method']):
    set_of_plot=set(plots)
    print(set_of_plot)
    fig = make_subplots(rows=3, cols=1, row_heights=[1,1,1],subplot_titles=("First Subplot","Second Subplot", "Third Subplot"))
        
    for _plot in range(len(plots)):
        print(_plot)
        X=plots[_plot]
        print(X)
        groups=set_of_plot.difference(set([X]))
        #groups=set(X).difference(set_of_plot)
        print(groups)
        x_axis_data = list(pd.unique(stats[X]))
        stats['concatenated'] = stats[groups].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        y_axis_data={}

        for index, row in stats.iterrows():
            key = 'concatenated'
            if row[key] not in y_axis_data:
                y_axis_data[row[key]] = []
            y_axis_data[row[key]].append(row[Y])

        bar=[]
        #print(y_axis_data)
        
        
        for group in y_axis_data:
            fig.add_trace(go.Bar(name=group, x=x_axis_data, y=y_axis_data[group],text=y_axis_data[group],textposition='outside'),row=_plot+1,col=1)

    #fig = go.Figure(data=bar)

    
    # Change the bar mode
    fig.update_layout(barmode='group',legend_title_text = "",height=1000, width=1500)
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="",range=(stats[Y].min()-0.05, stats[Y].max()+0.05))
    
    #fig.show()
    #fig.write_image("fig1.png")
    fig.write_html("index.html")

'''
    with open('index.html', 'w') as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='https://cdnjs.cloudflare.com/ajax/libs/plotly.js/1.33.1/plotly.min.js'))
'''