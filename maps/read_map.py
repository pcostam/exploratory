# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:57:51 2020

@author: anama
"""

import geopandas as gpd
import pandas as pd
from base.Network import Network

#Os sensores com o código RSV R5 estão associados à localização geográfica “Rotunda 5”
#No entanto, apenas os sensores  “RSV R5 Caudal Caixa” e o “RSV R5 Pressao caixa 2” estão afetos à rede de distribuição de água e, assim, com interesse para o estudo.
def mapeamento_scada():
    import pandas as pd
    import os
    import mysql.connector
    from sqlalchemy import create_engine
    path = os.path.join(os.getcwd(), "wisdom/maps/", "Correspond_Medidores_SIG_EPANET.xlsx" )
    flow = pd.read_excel(path, usecols="C:G", sheet_name="Medidores Telegestão", header=3, nrows=7)
    print("df columns", flow.columns)
    print("df head", flow)
    
    pressure = pd.read_excel(path, usecols="C:F", sheet_name="Medidores Telegestão", header=15, nrows=6)
    print("df columns", pressure.columns)
    print("df head", pressure)
    
    db_connection = 'mysql+pymysql://root:banana@localhost/infraquinta'
    conn = create_engine(db_connection)
    query = """ SELECT id, name
                FROM sensortg
            """
    
    df_sensors = pd.read_sql(query, conn)
    print("df columns", df_sensors.columns)
    print("df head", df_sensors)
    
    for index, row in df_sensors.iterrows():
        for _, row_pressure in pressure.iterrows():
            if row['name'] == row_pressure['Designação SCADA']:
                df_sensors.at[index,'Nó SIG'] = row_pressure['Nó SIG']
                df_sensors.at[index,'Conduta SIG']  = row_pressure['Conduta SIG']
        for _, row_flow in flow.iterrows():
             if row['name'] == row_flow['Designação SCADA']:
                df_sensors.at[index,'Nó SIG'] = row_flow['Nó SIG']
                df_sensors.at[index,'Conduta SIG']  = row_flow['Conduta SIG']
    df_sensors.dropna(inplace=True) 
    #amc sempre no
    #atu conduta
    print("df_sensors", df_sensors)
    return df_sensors


def get_infra(shape_name):
    import os
    path = os.path.join(os.getcwd(), "wisdom/maps/", shape_name )
    gdf = gpd.read_file(path)
    geom_col = gdf.geometry.name
  
    gdf = gdf[[geom_col, 'idinfraest']]
    print("gdf head", gdf.head())

    return gdf
       
    
def get_infraestruturas():
    gdf_concat = list()
    gdf_concat.append(get_infra("ramais.shp"))
    gdf_concat.append(get_infra("hidrantes.shp"))
    gdf_concat.append(get_infra("pontos_consumo.shp"))   
    gdf = pd.concat(gdf_concat, ignore_index=True)
    return gdf
    
def get_medidores_caudal():
    import os
    path = os.path.join(os.getcwd(), "wisdom/maps/","medidores_caudal.shp" )
    gdf = gpd.read_file(path)
    gdf = gdf[gdf['telegestao']=='ComTransmissao']
    return gdf
    
    

def get_medidores_pressao():
    import os
    path = os.path.join(os.getcwd(), "wisdom/maps/","medidores_pressao.shp" )
    gdf = gpd.read_file(path)
    gdf = gdf[gdf['telegestao']=='ComTransmissao']
    return gdf
    
    


def detectCloseSensor(src_points, candidates,  k_neighbors=1):
    from sklearn.neighbors import BallTree
  
    """
     
    Find nearest neighbors for all source points from a set of candidate points
    
     Parameters
     ----------
     src_points : sao infrasestruturas
     candidates: sao sensores
    
     Returns
     -------
     TYPE
         DESCRIPTION.
    
     """
 
    # Create tree from the candidate points
    print("candidates", candidates)
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc. indices[0] distances[0]
    closest = indices.ravel()
    closest_dist = distances.ravel()

    # Return indices and distances
    return (closest, closest_dist)
  
def get_candidates(left, right, map_scada, medidores):
    # Parse coordinates 
    #src points
    closest, closest_dist = detectCloseSensor(left, right)
    print("closest", closest)
    #repete numeros de closest - porque?
    closest_points = medidores.iloc[closest]
    print("CLOSEST POINTS", closest_points)
    sensor_ids = list()
    #identidade parece so ter ramais
    for infra in closest_points["idinfraest"]:
        try:
            line = map_scada[map_scada['Conduta SIG']==infra]
            if not(line.empty) and (len(sensor_ids) < 2):
                sensor_id = line["id"].iloc[0]
                sensor_ids.append(sensor_id)
        except KeyError:
            pass
   
    #idinfraest atu nó
   
    for infra in closest_points["identidade"]:
        try:
            line = map_scada[map_scada['Nó SIG']==infra]
            if not(line.empty) and (len(sensor_ids) < 5) :
                sensor_id = line["id"].iloc[0]
                sensor_ids.append(sensor_id)
        except KeyError:
            pass
    sensor_ids = list(set(sensor_ids))
    return sensor_ids
    
    
def read_events(events):
    import os
    import datetime
    import numpy as np
    path = os.path.join(os.getcwd(), "wisdom/maps/","Intervencoes_2017.xlsx" )
    df = pd.read_excel(path, sheet_name='Totais', header=1)
    infraestruturas = get_infraestruturas()  
    detected_infra = list()
    count_ab = 0
    map_scada = mapeamento_scada()
    for index, row in df.iterrows():
        if row['Infraestrutura'] in infraestruturas['idinfraest'].tolist():
            detected_infra.append(row['Infraestrutura'])
            if isinstance(row['Data abertura de água'], str):
                date = datetime.datetime.strptime(row['Data abertura de água'], '%Y-%m-%d %H:%M:%S')
                for event in events:
                    if date == event.getEnd():
                        count_ab += 1
                        sensor_ids = list()
                        # Parse coordinates 
                        #src points
                        #parece nao ser unico????
                        aux = infraestruturas[infraestruturas['idinfraest'] == row['Infraestrutura']]
                        left = np.array(list(aux.geometry.apply(lambda x: (x.centroid.x, x.centroid.y))))
                        #candidates
                        #filtrar aqui candidatos para sensores de caudais que estao para o estudo
                        right = np.array(list(get_medidores_caudal().geometry.apply(lambda x: (x.centroid.x, x.centroid.y))))
                        print("caudal", get_medidores_caudal().shape[0])
                        print("right", len(right))
                        sensor_ids = get_candidates(left, right, map_scada, get_medidores_caudal())
                        aux = infraestruturas[infraestruturas['idinfraest'] == row['Infraestrutura']]
                        left = np.array(list(aux.geometry.apply(lambda x: (x.centroid.x, x.centroid.y))))
                        #candidates
                        right = np.array(list(get_medidores_pressao().geometry.apply(lambda x: (x.centroid.x, x.centroid.y))))
                        sensor_ids += get_candidates(left, right, map_scada, get_medidores_pressao())
                        print("event id", event.getId())
                        print("sensor_ids", sensor_ids)
                   
                            
          
    print("count infra", len(detected_infra))
    print("count infra", detected_infra)
    print("count abertura", count_ab)
        
    
    return True

def test():
   
     network = Network("infraquinta", typeData="real", chosen_sensors=['12'],no_leaks=20, load=True)
     events = list(network.getEvents())
     print("events", events)
     read_events(events)
     

test()
