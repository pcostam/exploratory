# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 20:54:48 2020

@author: susan
"""

import pandas as pd
import sys
sys.path.append('../Functions')
from configuration import *
from epanettools.epanettools import EPANetSimulation, Node, Link, Network, Nodes, Links, Patterns, Pattern, Controls, Control # import all elements needed
from epanettools.examples import simple # this is just to get the path of standard examples
import configuration

path_init = "F:\\manual\\Tese\\exploratory\\wisdom\\dataset\\"
path = path_init + "INP Files\\StatusQuoInverno2018.inp"
es=EPANetSimulation(path)

nodes = [es.network.nodes[x].id for x in list(es.network.nodes)[:]]
links = [es.network.links[x].id for x in list(es.network.links)[:]]

print("Nodes: " + str(len(es.network.nodes)))
print("Links:" + str(len(es.network.links)))

files = ['node_pressure_summer', 'node_pressure_winter', 'link_flow_summer', 'link_flow_winter']

print("Export initiated")

for file in files:
    
    print("  " + file)
    
    path_import = path_init + "\\simulated\\original\\" + file + ".txt"
    path_export = path_init + "\\simulated\\" + file + ".csv"

    df = pd.read_csv(path_import, sep="  ", header=None)
    
    if ("node" in file):
        df.columns = nodes
    else:
        df.columns = links
    
    df = abs(df)
    
    df.to_csv(index=False, path_or_buf=path_export)

print("Export completed")