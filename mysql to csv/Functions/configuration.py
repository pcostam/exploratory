# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 20:31:44 2020

@author: susan

"""

import xml.etree.ElementTree as ET

def read_config():    
    path = "..\\config.xml"     
    doc = ET.parse(path)
    root = doc.getroot()           
    return root

def get_db(root): 
    host = root.find('database').get('host')
    user = root.find('database').get('user')
    pw = root.find('database').get('pw')
    db_config = {'host': host, 'user': user, 'pw': pw } 
    return db_config

def get_path(root):
    path = root.find('path').text 
    return path.replace("\\", "\\\\")

def get_wmes(root):   
    wmes = []
    for wme in root.findall('wme'):
        wmes.append(wme.get('name'))   
    return wmes

def get_wmes_sensors(root):
    wmes = []
    for wme in root.findall('wme'):
        wmes.append([wme.get('name'),int(wme.get('sensors'))])
    return wmes
