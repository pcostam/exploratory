# - coding: utf-8 --
'''
@info utilities to draw the ilu gui
@author Rui Henriques
@version 1.0
'''

import pandas as pd
import dash_core_components as dcc, dash_html_components as html
from dash.dependencies import State
from enum import Enum


''' ============================= '''
''' ====== A: LAYOUT UTILS ====== '''
''' ============================= '''

Button = Enum('Button', 'radio input checkbox multidrop daterange date time unidrop graph figure html link text upload')
colors = {'red':'#ed553b','yellow':'#f6d55c','dgreen':'#34988e','green':'#3caea3','blue':'#20639b',
          'lightblue':'#8fcccb','greenblue':'#449489','purple':'#4b3b9c','orange':'#d46e33','pale':'#fff4e0',
          'pink':'#e34262','wine':'#94353d','brown':'#9c656c','cream':'#d1b48c','lime':'#b4ba47'}
week_days = {'monday':1,'tuesday':2,'wednesday':3,'thursday':4,'friday':5,'saturday':6,'sunday':7}
calendar = {'all':range(1,8),'weekday':range(1,6),'weekend':range(6,8)}


def button(button_id, title, values, radio, sel_options=None):
    button = None
    style = {'display':'inline-block'}
    
    ### A: Input, DateRange buttons ###
    if radio is Button.input: 
        button = dcc.Input(id=button_id,value='0' if values is None else values,style={'width':'100%'})
    elif radio is Button.daterange: 
        button = dcc.DatePickerRange(id=button_id,start_date=pd.to_datetime(values[0]),end_date=pd.to_datetime(values[1]))
    elif radio is Button.date: 
        button = dcc.DatePickerSingle(id=button_id,date=pd.to_datetime(values))
    elif radio is Button.time: 
        button = dcc.Slider(id=button_id,min=0,max=24,step=0.5,value=int(values))
    elif radio is Button.upload: 
        button = dcc.Upload(id=button_id,
                            children=html.Div(['Drag and Drop or ',html.A('Select time series files')]),
                            style={'width':'90%','height':'60px','lineHeight':'60px','borderWidth':'1px',
                                   'borderStyle':'dashed','borderRadius':'5px','textAlign':'center','margin':'30px'},
                            multiple=True)
        
    ### B: HTML buttons ###
    elif radio is Button.html:
        button = html.Div(id=button_id,children=[values])
        if sel_options is None: return button
    elif radio is Button.text:
        textStyle = {'width':'58%','height':'300px','font-family':'monospace','font-size':'15px'}
        button = html.Div(style={'overflow':'auto','display':'flex','flex-direction':'column-reverse'},
                          children=[dcc.Textarea(id=button_id,value=values,placeholder=values, style=textStyle)])
    elif radio is Button.graph or radio is Button.figure: 
        import plot_utils
        button = dcc.Graph(id=button_id, figure=plot_utils.get_null_plot())

    ### C: Radio, CheckBox, Drop buttons ###
    else: 
        options = []
        for v in values: 
            if type(v) is str: options.append({'value':v, 'label':v.replace('_',' ').capitalize()})
            else: options.append({'value':str(v), 'label':str(v)})
        if radio is Button.radio: 
            sel_option = options[0]['value'] if sel_options is None else sel_options
            button = dcc.RadioItems(id=button_id,options=options,value=sel_option,labelStyle=style)
        elif radio is Button.checkbox: 
            if sel_options is None: sel_options = []
            button = dcc.Checklist(id=button_id,options=options,value=sel_options,labelStyle=style)
            if len(values)==1: return html.Div(button)
        else:
            multi = True if radio is Button.multidrop else False
            sel_options = [options[0]['value']] if sel_options is None else sel_options
            button = dcc.Dropdown(id=button_id,options=options,value=sel_options,multi=multi,style={'width':'100%'})
            
    ### D: return framed button ###
    title = title.replace('_',' ').capitalize()+':'
    return html.Div([html.Label(title,style={'margin-top':10,'font-size':'14px','font-weight':'bold'}),button])


def get_block_parameters(block_id, title, width, parameters, prefix="", hidden=False):
    boxstyle = {'background-color':'#dce7f3','width':('%f%%' % width),'border-radius':'5px',
                'border':'none','display':'inline-block','vertical-align':'top','padding':15,'margin':5}
    if hidden: boxstyle['display']='none'
    if title is None: block=[]
    else: block = [html.Label(title.replace('_',' ').capitalize(),style={'font-weight':'bold','font-style':'italic'})]
    for param in parameters: 
        sel_options = None if len(param)<=3 else param[3]
        block.append(button(prefix+param[0],param[0],param[1],param[2],sel_options))
    return html.Div(block,id=block_id,style=boxstyle)


def get_hidden_components(hidden_parameters, width, qprefix):
    hidden_components = []
    for key in hidden_parameters:
        block = get_block_parameters(qprefix+key,key.capitalize()+" parameters",width,hidden_parameters[key],qprefix+key,True)
        hidden_components.append(block)
    return hidden_components


def get_layout(pagetitle, parameters, visuals, hidden_components=[], prefix=""):
    '''A: create buttons'''
    titlestyle = {'color':'#cc6666','margin':6,'vertical-align':'bottom','margin-bottom':12,'margin-left':10,'margin-top':10,'font-weight':'bold','display':'inline-block'}
    layout = [html.H6(pagetitle, style=titlestyle)]
    layout.append(html.Br())
    #prefix = "query_param_all" if not hidden_components else ""
    for blockparam in parameters: 
        layout.append(get_block_parameters(prefix+blockparam[0],blockparam[0],blockparam[1],blockparam[2],prefix))
    layout+=hidden_components
        
    '''B: finish layout'''    
    submitstyle = {'background-color':'#6ABB97','border':'none','font-size':'14px','width':'58%','margin':5,'margin-top':20,'margin-bottom':25}
    layout.append(html.Button('Run query', id='button',style=submitstyle))
    for param in visuals: 
        layout.append(button(prefix+param[0],param[0],param[1],param[2],None if len(param)<=3 else param[3]))
    return html.Div(layout,id='subroot',style={'width':'95%','margin':50})


''' ================================ '''
''' ====== B: OTHER APP UTILS ====== '''
''' ================================ '''

def get_states(parameters, skipdates=False, prefix=""):
    result = []
    if not skipdates: result = [State(prefix+parameters[0][0],'start_date'),State(prefix+parameters[0][0],'end_date')]
    start = 0 if skipdates else 1
    for i in range(start,len(parameters)): 
        result.append(State(prefix+parameters[i][0],'value'))
    return result

def get_null_label():
    return html.Label('None',style={'color':'gray'})

def get_calendar_days(calendario):
    dias = set()
    for entry in calendario:
        if entry in week_days: dias.add(week_days[entry])
        elif entry in calendar: dias.update(calendar[entry])
    if len(dias)==7: return "all"
    return list(dias)
