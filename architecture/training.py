# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:26:20 2020

@author: anama
"""

from architecture.autoencoder import autoencoderLSTM
from architecture.cnn_lstm import CNN_LSTM
from architecture.cnn_biLSTM import CNN_BiLSTM
from architecture.scb_lstm import SCB_LSTM
from architecture.stacked_lstm import stacked_LSTM
from architecture.stacked_bilstm import stacked_BiLSTM
from architecture.parameters import user_parameters

def do_parameters(data):
    user_parameters = user_parameters(dropout=data['dropout'], n_steps=data['n_steps'],
                                 n_input=data['n_input'],  n_seq=data['n_seq'], regularizer=data['regularizer'], n_train=data['n_train']
                                 ,simulated = data['simulated'],
                                 n_leaks=data['n_leaks'], save=data['save'],
                                 validation=data['validation'])

    return user_parameters

models = ["autoencoder LSTM", "CNN-BiLSTM", "CNN-LSTM", "SCB-LSTM", "stacked BiLSTM", "stacked LSTM"]
def training(model_name ="all", type_model = "multi-channel", timesteps=96, simulated = False, bayesian=False, save=True, validation=True, hidden_size=16, code_size=4):
    
    data = pd.read_csv("user_parameters.csv") 
    user_parameters = do_parameters(data)
    
    if model_name == "all":
        for model in models:
            print("dotrain")
            
    if model_name == "autoencoder LSTM":
         autoencoder = autoencoderLSTM()
         autoencoder.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, hidden_size=hidden_size, code_size=code_size)
    
    elif model_name == "CNN-LSTM":
         cnn_lstm = CNN_LSTM(type_model=type_model, model_name="CNN-LSTM")
         cnn_lstm.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, validation=validation)
    
    elif model_name == "CNN-BiLSTM":
         cnn_lstm = CNN_LSTM(type_model=type_model, model_name="CNN-BiLSTM")
         cnn_lstm.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, validation=validation)
 
    elif model_name == "SCB-LSTM":
          cnn_lstm = CNN_LSTM(type_model=type_model, model_name="SCB-LSTM")
          cnn_lstm.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, validation=validation)
    
    elif model_name == "stacked LSTM":
          stacked_lstm = stacked_LSTM()
          stacked_lstm.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, validation=validation)

    elif model_name == "stacked BiLSTM":
          stacked_bilstm = stacked_BiLSTM()
          stacked_bilstm.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, validation=validation)
    else:
        raise ValueError("No such architecture")