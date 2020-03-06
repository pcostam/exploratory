'''
@info utilities to draw the plots
@author Rui Henriques
@version 1.0
'''

import numpy as np
import plotly.tools as tls, plotly.figure_factory as plt, plotly.graph_objs as go

''' ================================= '''
''' ====== A: LINE CHART UTILS ====== '''
''' ================================= '''

def get_series_plot(series,title):
    
    '''A: chart lines'''
    fig = tls.make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.00000001, horizontal_spacing=0.001)
    for col in series.columns:
        fig.append_trace({'x':series.index, 'y':series[col], 'type':'scatter', 'name':col}, 1, 1)
        
    '''B: chart layout'''
    fig['layout'].update(dict(height=900,barmode='group',yaxis=dict(title=title),
                  xaxis=dict(title='tempo',autorange=True,rangeslider=dict(visible=True),tickangle=45,
                             rangeselector=dict(buttons=list([dict(step='all'),
                                     dict(stepmode='backward',step='hour',count=12,label='12 Horas',visible=True),
                                     dict(count=1,stepmode='backward',step='day',label='1 Dia',visible=True),
                                     dict(count=3,stepmode='backward',step='day',label='3 Dias',visible=True)])))))
    return fig

def add_predictor_series(fig, predictor):
    for var in predictor.variables:
        fig.append_trace(go.Scatter(name='Model['+var+']', x=predictor.index, yaxis='y1', y=predictor.series[var]["model"], mode='lines'),1,1)
        fig.append_trace(go.Scatter(name='Upper Bound['+var+']', x=predictor.index, yaxis='y1', y=predictor.series[var]["upperbound"], line=dict(color='rgb(68,68,68,0.2)', width=2, dash='dash')),1,1)
        fig.append_trace(go.Scatter(name='Lower Bound['+var+']', x=predictor.index, yaxis='y1', y=predictor.series[var]["lowerbound"], fill="tonexty", fillcolor='rgba(68,68,68,0.2)', line=dict(color='rgb(68,68,68,0.2)', width=2, dash='dash')),1,1)

def add_anomalies(fig, series, predictor):
    for var in predictor.variables:
        anomalies = np.array([np.NaN]*len(series[var]))
        lowerpositions = series[var].values<predictor.series[var]["lowerbound"]
        upperpositions = series[var].values>predictor.series[var]["upperbound"]
        anomalies[lowerpositions] = series[var].values[lowerpositions]
        anomalies[upperpositions] = series[var].values[upperpositions]
        fig.append_trace({'x':series.index, 'y':anomalies, 'yaxis':'y1', 'mode':'markers', 'name':'Anomalias['+var+']'},1,1)
        fig.update_traces(marker=dict(size=12,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))

def get_null_plot(message=None):
    title='parameterize and click <b>run</b> to visualize'
    if message is not None: title=message
    nulllayout = go.Layout(height=200, title=title, font=dict(size=9)) #color='#7f7f7f'
    return go.Figure(layout=nulllayout) 

''' ============================ '''
''' ====== B: CORRELOGRAM ====== '''
''' ============================ '''
    
def get_correlogram(series):
    x = []
    for col in series.columns: x.append(col)
    z = series.corr()
    z = z.round(2).values.tolist()
    corr = plt.create_annotated_heatmap(z=z, x=x, y=x, hoverinfo='z', colorscale='Reds')
    corr.layout.margin.l=300
    corr.layout.update(go.Layout(width=800+200*len(x),height=150+100*len(x)))
    return corr