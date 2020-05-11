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


models = ["autoencoder LSTM", "CNN-BiLSTM", "CNN-LSTM", "SCB-LSTM", "stacked BiLSTM", "stacked LSTM"]
def training(type_model="all", timesteps=96, simulated = False, bayesian=False, save=True, validation=True, hidden_size=16, code_size=4, cnn_lstm_type = "multi-channel"):
    if type_model == "all":
        for model in models:
            print("dotrain")
            #do_train
    if type_model == "autoencoder LSTM":
         autoencoder = autoencoderLSTM()
         autoencoder.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, hidden_size=hidden_size, code_size=code_size)
    
    elif type_model == "CNN-LSTM":
         cnn_lstm = CNN_LSTM(cnn_lstm_type)
         cnn_lstm.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, validation=validation)
    
    elif type_model == "CNN-BiLSTM":
         cnn_lstm = CNN_LSTM(cnn_lstm_type, decoder="LSTM")
         cnn_lstm.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, validation=validation)
    
         #cnn_bilstm = CNN_BiLSTM()
         #cnn_bilstm.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, validation=validation)
    
    elif type_model == "SCB-LSTM":
          scb_lstm = SCB_LSTM()
          scb_lstm.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, validation=validation)
      
    elif type_model == "stacked LSTM":
          stacked_lstm = stacked_LSTM()
          stacked_lstm.do_train(timesteps=3, simulated = simulated, bayesian=bayesian, save=save, validation=validation)

    elif type_model == "stacked BiLSTM":
          stacked_bilstm = stacked_BiLSTM()
          stacked_bilstm.do_train(timesteps=3, simulated = simulated, bayesian=bayesian, save=save, validation=validation)
    else:
        raise ValueError("No such architecture")