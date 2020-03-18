'''
@info gui global variables: dash app and database connector
@author Rui Henriques
@version 1.0
'''

import dash
import os
assets_path = os.getcwd() +'//assets'

external_stylesheets = [assets_path]
app = dash.Dash(__name__, assets_folder = assets_path, include_assets_files = True)  
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True