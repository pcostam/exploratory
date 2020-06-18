# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:26:20 2020

@author: anama
"""

from architecture.autoencoder import autoencoderLSTM
from architecture.cnn_lstm import CNN_LSTM
from architecture.stacked_lstm import stacked_LSTM
from architecture.stacked_bilstm import stacked_BiLSTM
from architecture.user_parameters import user_parameters

def do_parameters(data):
    param = user_parameters(dropout=data['dropout'].iloc[0]
                                 ,n_steps=data['n_steps'].iloc[0]
                                 ,n_seq=data['n_seq'].iloc[0]
                                 ,regularizer=data['regularizer'].iloc[0]
                                 ,n_train=data['n_train'].iloc[0]
                                 ,simulated = data['simulated'].iloc[0]
                                  ,n_leaks=data['n_leaks'].iloc[0]
                                 ,save=data['save'].iloc[0]
                                 ,validation=data['validation'].iloc[0]
                                 ,bayesian=data['bayesian'].iloc[0]
                                 ,hidden_size=data['hidden_size'].iloc[0]
                                 ,code_size=data['code_size'].iloc[0])

    return param

models = ["autoencoder LSTM", "CNN-BiLSTM", "CNN-LSTM", "SCB-LSTM", "stacked BiLSTM", "stacked LSTM"]
def training(model_name ="all", type_model = "multi-channel", n_test=1):
    import pandas as pd
    import os
    print("cwd", os.getcwd())
    path = os.path.join(os.getcwd(), "wisdom/architecture/","user_parameters.csv" )
    data = pd.read_csv(path, delimiter=';') 
    data = data[data['n_test'] == n_test]
    user_parameters = do_parameters(data)
  
    
    if model_name == "all":
        for model in models:
            print("dotrain")
            
    if model_name == "autoencoder LSTM":
         autoencoder = autoencoderLSTM()
         autoencoder.do_train(user_parameters)
    
    elif model_name == "CNN-LSTM":
         cnn_lstm = CNN_LSTM(type_model=type_model, model_name="CNN-LSTM")
         #cnn_lstm.do_train(user_parameters)
         cnn_lstm.do_test(user_parameters)
    
    elif model_name == "CNN-BiLSTM":
         cnn_lstm = CNN_LSTM(type_model=type_model, model_name="CNN-BiLSTM")
         cnn_lstm.do_train(user_parameters)
       
 
    elif model_name == "SCB-LSTM":
          cnn_lstm = CNN_LSTM(type_model=type_model, model_name="SCB-LSTM")
          cnn_lstm.do_train(user_parameters)
    
    elif model_name == "stacked LSTM":
          stacked_lstm = stacked_LSTM()
          stacked_lstm.do_train(user_parameters)

    elif model_name == "stacked BiLSTM":
          stacked_bilstm = stacked_BiLSTM()
          stacked_bilstm.do_train(user_parameters)
    else:
        raise ValueError("No such architecture")
        
training(model_name="CNN-LSTM", n_test=1)