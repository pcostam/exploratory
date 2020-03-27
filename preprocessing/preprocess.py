# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:01:51 2020

@author: anama
"""
from preprocessing.series import preprocess_data
def preprocess(data, granularity, start_date, end_date):
    data = preprocess_data(data, granularity, start_date, end_date)    
    return data
    