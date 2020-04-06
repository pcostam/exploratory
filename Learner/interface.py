# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:55:08 2020

@author: anama
"""

from architecture import autoencoder
from preprocessing import preprocess
def operation(data, method, start_date, end_date, granularity, anomaly_threshold):
    #preprocessing data according to granularity and time period
    prediction = list()
    
    print("head data", data.head())
    data = preprocess.preprocess_data(data, granularity, start_date, end_date)
    print("data after preprocess", data.head())
    #Model
    if method == "LSTM autoencoders":
        print("Operation LSTM autoencoder")
        return autoencoder.autoencoderLSTM.operation(data, anomaly_threshold)
    #elif method == "CNN-LSTM":
    #elif method == "CNN-Bi-LSTM":
    #elif method == "stacked Bi-LSTM":
    #elif method == "SCB-LSTM":
        
        
    
    #prediction with anomaly scores
    return prediction