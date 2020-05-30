# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:30:04 2020

@author: anama
"""

import pandas as pd
import datetime
from epanettools.epanettools import EPANetSimulation, Node, Link, Network, Nodes, Links, Patterns, Pattern, Controls, Control # import all elements needed
from epanettools.examples import simple # this is just to get the path of standard examples
import re

pressure_base_filename =  "Rotura_Pnova"
real_pressure_base_filename = "Pressao_SensorMeioRede"
flow_base_filename = "Rotura_Q_Medidores"
#nodes ==> pressure
epanet_to_scada_pressure = {'aMI817150114':3,
                            'aMI817150114':7,
                             'aMC402150114':8, 
                             'aMC404150114':11,
                             'aMC401150114':13,
                             'aMC403150114':15}

#flow ==> pipes/links
epanet_to_scada_flow = {
                        '6': 1,
                        'aTU4981150302':10,
                        'aTU1096150205': 2,
                        'aTU455150205':9, 
                        'aTU1477150205':14,
                        '2': 12,
                        'aTU1093150205':6}

no_leaks = 18696
no_flow_files = 18696
no_pressure_files = 18696
no_pressure_sensors = 21
no_flow_sensors = 7


import os


path_init = "F:\\manual\\Tese\\exploratory\\wisdom\\dataset\\"

path = path_init + "INP Files\\StatusQuoVerao2018.inp"
es=EPANetSimulation(path)

tg_links = [es.network.links[x].id for x in list(es.network.links)[:] if es.network.links[x].id in epanet_to_scada_flow.keys()]
print(tg_links)
#7
print("Links:" + str(len(tg_links)))

from os import listdir
from os.path import isfile, join
pathtxt = path_init + "\\simulated\\fugas_txt\\"
pathflowtxt =  path_init + "\\simulated\\fugas_txt\\Rotura_Q\\"
pathpressuretxt = path_init + "\\simulated\\fugas_txt\\Rotura_Pnova\\"
files = []
#files = [f for f in listdir(pathtxt) if isfile(join(pathtxt, f))]

flowfiles = [f for f in listdir(pathflowtxt) if isfile(join(pathflowtxt, f))]
flow_files = [namefile for namefile in flowfiles if flow_base_filename in namefile]

pressurefiles = [f for f in listdir(pathpressuretxt) if isfile(join(pathpressuretxt, f))]
pressure_files = [namefile for namefile in pressurefiles if pressure_base_filename in namefile]

files += flowfiles
#files += pressurefiles  
print("Export initiated")


def write_format_leaks(df, no_leak, path_init, new_filename):
    print("Filename:", new_filename)
    print("Id leak:", no_leak)
    rows = df.shape[0]
    total_seconds = rows*600
    times = [x for x in range(0,total_seconds,600)]
    df["time"] = times
    df["leak"] = 0
    df = abs(df)
    path_map = path_init + "\\simulated\\mapeamento_fugas\\TabelaArquivoFinal.xlsx"
    df_map = pd.read_excel(path_map)
    base = "Rotura_P"
    filename = base + str(no_leak) + ".txt"
   
    if 'Arquivo' in df_map.columns:
        df_map = df_map[(df_map['Arquivo']) == filename]
    else:
        raise ValueError("Column Arquivo not in excel")
        
    stime = df_map["Tempo inicial (seg)"].iloc[0]
    etime = df_map["Tempo final(seg)"].iloc[0]

    for index, row in df.iterrows():
        time = row['time']
        if(time >= stime) and (time <= etime):
            df['leak'].iloc[index] = 1    
            
    path_export = path_init + "\\simulated\\" + folder + "\\" + new_filename + str(no_leak) + ".csv"
    #Export to csv file
    df.to_csv(index=False, path_or_buf=path_export)
    
#1 ficheiro txt com os dados das 18696 roturas,
# em que cada linha corresponde a um time step de 600 segundos (10 minutos), 
#começando no t=0, e cada coluna a uma rotura.
i = 0
for file in files[10775:]:
    i += 1
    try:
        print("  " + str(file))
        
        path_import = path_init + "\\simulated\\fugas_txt\\" + file 
        
       
       
        
        no_leak = 0
        folder = ""
        new_filename = ""
        base_filename = ""
        """
        if (pressure_base_filename in file):
            #does not correspond to real sensors
            base_filename = pressure_base_filename
            folder = "fugas_P"
            new_filename = "pressure_leak"
            no_leak = re.search(r'\d+', file).group()
            write_format_leaks(df, no_leak, path_init, new_filename)
            print("end")
        """       
        if (flow_base_filename in file):
          path_import = path_init + "\\simulated\\fugas_txt\\Rotura_Q\\" + file 
          df = pd.read_csv(path_import, sep="  ", header=None)
          map_id = epanet_to_scada_flow
          #transform INP name to EPANET name
          #corresponds to real sensor
          new_tg_links = [map_id[name] for name in tg_links]
          df.columns = new_tg_links
          base_filename = flow_base_filename
          folder = "fugas_Q"
          new_filename = "flow_leak"
          no_leak = re.search(r'\d+', file).group()
          print("no leak", no_leak)
          write_format_leaks(df, no_leak, path_init, new_filename)
          print("end")
          
        """
        elif (real_pressure_base_filename in file):
            #each column is a leak and not a sensor. 
            #Line is a sensor - a middle pressure sensor
            base_filename = real_pressure_base_filename 
            folder = "fugas_P_real"
            new_filename = "P_real"
            #write each leak in a file
            no_leak = 1
            for column in df.columns:   
                new = df.filter([column], axis=1)
                no_leak += 1
                write_format_leaks(new, no_leak, path_init, new_filename)
            print("end")
        """
    except:
            f = open("failed_simulated_leaks.txt", "w")
            print("Failed to Run")
            f.write('Failed to run index {}\n'.format(i))
            f.close()
            break
   


print("Export completed")




        
            
    
    