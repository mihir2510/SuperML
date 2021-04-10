import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots


def bar_2d(stats, Y, X, groups,file_name='index',download_png=None,height=None,width=None):
    '''
    Plots the data given as input and saves it as an PNG and HTML files

            Parameters:
                    stats (pandas dataframe) : Data to be plotted
                    Y (string) : The parameter / column name of metric to be plotted on Y Axis
                    X (string) : The parameter / column name of metric to be plotted on X Axis
                    groups (list) : List of strings containing the column names to be grouped 
                    file_name (string) : Name for the HTML file to be saved
                    download_png (boolean) : Do you want an png file for the plot
                    height (integer) : height of the plot
                    width (integer) : width of the plot


    '''


    x_axis_data = list(pd.unique(stats[X]))
    stats['concatenated'] = stats[groups].apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
    y_axis_data={}

    for index, row in stats.iterrows():
        key = 'concatenated'
        if row[key] not in y_axis_data:
            y_axis_data[row[key]] = []
        y_axis_data[row[key]].append(row[Y])

    bar=[]
    for group in y_axis_data:
        bar.append(go.Bar(name=group, x=x_axis_data, y=y_axis_data[group],text=y_axis_data[group],textposition='outside',hovertemplate=y_axis_data[group]))

    fig = go.Figure(data=bar)
    
    # Change the bar mode
    fig.update_layout(title='Bar Plot',barmode='group',legend_title_text = "Legend",hovermode="closest",height=height, width=width) #showlegend=False,
    fig.update_xaxes(title_text=X)

    fig.update_yaxes(title_text=Y,range=(stats[Y].min()-0.05, stats[Y].max()+0.05))

    
    #fig.show()
    if download_png:
        fig.write_image(download_png+".png")
        print('PNG File Generated')

    fig.write_html(file_name+".html")
    print('HTML File Created')


def bar_2dsubplot(stats, Y, plots,file_name='index',download_png=None,height=None,width=None):
    '''
    Plots the data given as input and saves it as an PNG and HTML files

            Parameters:
                    stats (pandas dataframe) : Data to be plotted
                    Y (string) : The parameter / column name of metric to be plotted on Y Axis
                    plots (list) : List of string denoting the columns to be used for generating subplots
                    file_name (string) : Name for the HTML file to be saved
                    download_png (boolean) : Do you want an png file for the plot
                    height (integer) : height of the plot
                    width (integer) : width of the plot


    '''
    
    set_of_plot=set(plots)
    #print(set_of_plot)
    fig = make_subplots(rows=3, cols=1, row_heights=[1,1,1],subplot_titles=plots)
        
    for _plot in range(len(plots)):
        
        X=plots[_plot]
        
        groups=set_of_plot.difference(set([X]))
        #groups=set(X).difference(set_of_plot)
        
        x_axis_data = list(pd.unique(stats[X]))
        stats['concatenated'] = stats[groups].apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
        y_axis_data={}

        for index, row in stats.iterrows():
            key = 'concatenated'
            if row[key] not in y_axis_data:
                y_axis_data[row[key]] = []
            y_axis_data[row[key]].append(row[Y])

        bar=[]
        #print(y_axis_data)
        
        
        for group in y_axis_data:
            fig.add_trace(go.Bar(name=group, x=x_axis_data, y=y_axis_data[group],text=y_axis_data[group],textposition='outside',hovertemplate=y_axis_data[group]),row=_plot+1,col=1)

    #fig = go.Figure(data=bar)

    
    # Change the bar mode
    fig.update_layout(title='Bar Plot',barmode='group',legend_title_text = "Legend",height=height, width=width,hovermode="closest")
    for _plot in range(len(plots)):
        fig.update_xaxes(title_text=plots[_plot],row=_plot+1,col=1)
        fig.update_yaxes(title_text=Y,range=(stats[Y].min()-0.05, stats[Y].max()+0.05),row=_plot+1,col=1)


    #fig.show()
    if download_png:
        fig.write_image(download_png+".png")
        print('PNG File Generated')

    fig.write_html(file_name+'.html')
    print('HTML File Created')
