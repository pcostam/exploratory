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
        ('granularity_(minutes)','15',gui.Button.input)]
processing_parameters = [
        ('mode',['default','parametric','fully_automatic'],gui.Button.radio,'default'),
        ('imputation_mode',['univariate','multivariate'],gui.Button.checkbox),
        ('imputation_method',['mean','amelia'],gui.Button.unidrop), 
        ('imputation_parameterization','<parameters here>',gui.Button.input), 
        ('assess_missings',['method_1','method_2'],gui.Button.unidrop), 
        ('outliers_mode',['point','subsequence'],gui.Button.checkbox),
        ('outliers_method',['method_A','method_B'],gui.Button.unidrop), 
        ('outliers_parameterization','<parameters here>',gui.Button.input), 
        ('assess_outliers',['method_1','method_2'],gui.Button.unidrop),
        ('evaluation metrics',["all"],gui.Button.unidrop,["all"])]

parameters = [('Target time series',27,target_options),('Processing options',27,processing_parameters)]
charts = [('visualizacao',None,gui.Button.figure),('correlograma',None,gui.Button.graph)]

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
    idate, fdate = pd.to_datetime(states['date.start_date']), pd.to_datetime(states['date.end_date'])    
    minutes = int(states['granularidade_em_minutos.value'])
    dias = [gui.get_calendar_days(states['calendario.value'])]
    
    '''B: retrieve data'''
    #data, name, = query_metro.retrieve_data(idate,fdate,contagem,dias,estacoes_entrada,estacoes_saida,minutes,["record_count"])'''
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
    corr = plot_utils.get_correlogram(series)

    '''C: Plot statistics here'''

    return fig, corr


''' ===================== '''
''' ====== C: MAIN ====== '''
''' ====================== '''

if __name__ == '__main__':
    app.layout = layout
    app.run_server()
