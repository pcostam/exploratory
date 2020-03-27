# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:46:15 2020

@author: anama
"""

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html

"""
that the app.layout describes what the app looks like 
and is a hierarchical tree of components. 
The dash_html_components library provides classes for all of the HTML tags, 
and the keyword arguments describe the HTML attributes like style, className, and id. 
The dash_core_components library generates higher-level components like controls and 
graphs.
"""
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=False)