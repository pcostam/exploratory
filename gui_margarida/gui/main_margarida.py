# - coding: utf-8 --
'''
@info webpage for predictive analysis of metro data
@author Inês Leite and Rui Henriques
@version 1.0
'''

import pandas as pd, dash
from gui.app import app
from gui import gui_utils as gui, plot_utils
from dash.dependencies import Input, Output

''' ================================ '''
''' ====== A: WEB PAGE PARAMS ====== '''
''' ================================ '''

pagetitle = 'WISDOM'

target_options = [
        ('upload',None,gui.Button.upload),
        ('sensor_type',["all"],gui.Button.multidrop,["all"]), 
        ('sensor_name',["all"],gui.Button.multidrop,["all"]), 
        ('period',['2018-10-02','2018-10-11'],gui.Button.daterange),
        ('calendar',list(gui.calendar.keys())+list(gui.week_days.keys()),gui.Button.multidrop,['all']),
        ('time_sampling_(seconds)','60',gui.Button.input)]
processing_parameters = [
        ('mode',['default','parametric'],gui.Button.radio,'default'),
        ('approach',["supervised_point_score", "time_to_outlier", "unsupervised_point_score"],gui.Button.unidrop), 
        ('method',["LSTM autoenconders", "CNN-LSTM", "CNN-Bi-LSTM", "stacked Bi-LSTM", "SCB-LSTM"],gui.Button.unidrop), 
        ('enhacement',['calendric-based_correction','weather-based_correction'],gui.Button.checkbox), 
        ('outlier threshold [0-100]','50',gui.Button.input),
        ('parameterization','<parameters here>',gui.Button.input), 
        ('assessment',['rolling_CV_(default)','rolling_CV_(with_Bayesian_optimization)'],gui.Button.unidrop,'rolling_CV_(with_Bayesian_optimization)'),
        ('evaluation_statistics',["all","learning_statistics","test_statistics","none"],gui.Button.multidrop,["all"])]

parameters = [('Target time series',27,target_options),('Processing options',27,processing_parameters)]
charts = [('visualizacao',None,gui.Button.figure),('results','Statistics on results here...',gui.Button.text)]

layout = gui.get_layout(pagetitle,parameters,charts)

def get_states():
    return gui.get_states(target_options+processing_parameters)

def agregar(mlist):
    agregado = set()
    for entries in mlist: 
        for entry in entries: 
            agregado.add(entry) 
    return list(agregado)

''' ============================== '''
''' ====== B: CORE BEHAVIOR ====== '''
''' ============================== '''

def get_data(states, series=False):
    
    '''A: process parameters'''
    idate, fdate = pd.to_datetime(states['period.start_date']), pd.to_datetime(states['period.end_date'])    
    minutes = int(states['time_sampling_(seconds).value'])
    dias = [gui.get_calendar_days(states['calendar.value'])]
    
    '''B: retrieve data'''
    #data, name, = retrieve_data(idate,fdate,contagem,dias,estacoes_entrada,estacoes_saida,minutes,["record_count"])'''
    return None #series_utils.fill_time_series(data,name,idate,fdate,minutes)


@app.callback([Output('visualizacao','figure'),Output('correlograma', 'figure')],
              [Input('button','n_clicks')])
def update_charts(inp,*args):
    if inp is None: 
        nullplot = plot_utils.get_null_plot()
        return nullplot, nullplot
    states = dash.callback_context.states
    #print(states)
    series = get_data(states,series=True)

    '''A: Run preprocessing here'''
    
    '''B: Plot time series and correlogram'''
    fig = plot_utils.get_series_plot(series,'titulo aqui')

    '''C: Plot statistics here'''

    return fig, corr


''' ===================== '''
''' ====== C: MAIN ====== '''
''' ====================== '''

if __name__ == '__main__':
    app.layout = layout
    app.run_server()
