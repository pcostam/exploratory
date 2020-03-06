'''
@info gui global variables: dash app and database connector
@author Rui Henriques
@version 1.0
'''

import dash

app = dash.Dash(__name__, assets_folder = 'assets', include_assets_files = True) 
server = app.server
app.config.suppress_callback_exceptions = True