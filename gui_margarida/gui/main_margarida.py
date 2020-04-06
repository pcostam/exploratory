# - coding: utf-8 --
'''
@info webpage for predictive analysis of metro data
@author Inês Leite and Rui Henriques
@version 1.0
'''

import pandas as pd, dash
import gui_utils as gui, plot_utils
from app import app
#from gui.app import app
#from gui import gui_utils as gui, plot_utils

#from gui_utils import * 
#from plot_utils import *
from dash.dependencies import Input, Output, State
#import dash_html_components as html
#import dash_core_components
import io
import base64
import Learner.interface as Learner

''' ================================ '''
''' ====== A: WEB PAGE PARAMS ====== '''
''' ================================ '''

pagetitle = 'WISDOM'

target_options = [
        ('upload',None,gui.Button.upload),
        ('sensor_type',["all"],gui.Button.multidrop,["all"]), 
        ('sensor_name',["all"],gui.Button.multidrop,["all"]), 
        ('period',['2017-01-01','2017-03-01'],gui.Button.daterange),
        ('calendar',list(gui.calendar.keys())+list(gui.week_days.keys()),gui.Button.multidrop,['all']),
        ('time_sampling_(seconds)','60',gui.Button.input)]
processing_parameters = [
        ('mode',['default','parametric'],gui.Button.radio,'default'),
        ('approach',["supervised_point_score", "time_to_outlier", "unsupervised_point_score"],gui.Button.unidrop), 
        ('method',["LSTM autoencoders", "CNN-LSTM", "CNN-Bi-LSTM", "stacked Bi-LSTM", "SCB-LSTM"],gui.Button.unidrop), 
        ('enhacement',['calendric-based_correction','weather-based_correction'],gui.Button.checkbox), 
        ('outlier threshold [0-100]','50',gui.Button.input),
        ('parameterization','<parameters here>',gui.Button.input), 
        ('assessment',['rolling_CV_(default)','rolling_CV_(with_Bayesian_optimization)'],gui.Button.unidrop,'rolling_CV_(with_Bayesian_optimization)'),
        ('evaluation_statistics',["all","learning_statistics","test_statistics","none"],gui.Button.multidrop,["all"])]

parameters = [('Target time series',27,target_options),('Processing options',27,processing_parameters)]
charts = [('visualizacao',None,gui.Button.figure),('results','Statistics on results here...',gui.Button.text)]

layout = gui.get_layout(pagetitle,parameters,charts)

def get_states():
    states = gui.get_states(target_options+processing_parameters)
    states.append(State('upload', 'contents'))
    states.append(State('upload', 'filename'))
    return states


states = get_states()
print("states", states)

def agregar(mlist):
    agregado = set()
    for entries in mlist: 
        for entry in entries: 
            agregado.add(entry) 
    return list(agregado)

''' ============================== '''
''' ====== B: CORE BEHAVIOR ====== '''
''' ============================== '''

def get_data(df, states, series=False):
    print(">>>get_data")
    '''A: process parameters'''
    start = pd.to_datetime(states['period.start_date'])
    end = pd.to_datetime(states['period.end_date'])
    granularity = int(states['time_sampling_(seconds).value'])
    anomaly_threshold = float(states['outlier threshold [0-100].value'])
    method = str(states['method.value'])
    #dias = [gui.get_calendar_days(states['calendar.value'])]
    
    print("granularity", granularity)
    print("start", start)
    print("method", method)
    print("thresh", anomaly_threshold)
  
    #print("dias", dias)
    print("Learner operation")
    return Learner.operation(df, method, start, end, granularity, anomaly_threshold)
    
    '''B: retrieve data'''
    #data, name, = retrieve_data(idate,fdate,contagem,dias,estacoes_entrada,estacoes_saida,minutes,["record_count"])'''
    #series_utils.fill_time_series(data,name,idate,fdate,minutes)

def retrieve_data(idate, fdate, contagem, dias, estacoes_entrada, estacoes_saida, minutes, record):
    
    #parse_contents()
    return True


#returns dataframe
def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), index_col=False)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter = r'\s+')
    except Exception as e:
        print(e)
        return ['There was an error processing this file.']
    
    
    print("df columns", df.columns)
    print("df index", df.index)
    return df

print("states", states)

@app.callback(Output('visualizacao', 'figure'), 
              [Input('button', 'n_clicks')],
              states)   
def update_graph(n_clicks, *args):
    print("update graph")
    
    states = dash.callback_context.states
   
    fig = list()
    data = list()
    contents = states['upload.contents'][0]
    if contents is not None:
        filename = states['upload.filename'][0]
        print("filename:", filename)
        df = parse_data(contents, filename)
        print(df.head())
        df['date'] = pd.to_datetime(df['date'])
        print("type", type(df))
        print("init df.columns", df.columns)
        print("init df.index", df.index)
        anomalies = get_data(df, states, series=False)
        print("anomalies", anomalies)
        df.index = df['date']
        df = df.drop(['date'], axis = 1)
        series = df
        fig = plot_utils.get_series_plot(series,'titulo aqui')
        plot_utils.add_anomalie_scores(fig, anomalies)
    return fig


"""
@app.callback([Output('visualizacao','figure'),Output('correlograma', 'figure')],
              [Input('button','n_clicks')])
def update_charts(inp,*args):
    print(">>>>update")
    for title in args:
        print('element: ', title)
        
    if inp is None: 
        nullplot = plot_utils.get_null_plot()
        return nullplot, nullplot
    states = dash.callback_context.states
    print(states)
    series = get_data(states,series=True)

    '''A: Run preprocessing here'''
    
    '''B: Plot time series and correlogram'''
    fig = plot_utils.get_series_plot(series,'titulo aqui')

    '''C: Plot statistics here'''

    return fig, corr

"""

''' ===================== '''
''' ====== C: MAIN ====== '''
''' ====================== '''

if __name__ == '__main__':
    app.layout = layout
    app.run_server()
