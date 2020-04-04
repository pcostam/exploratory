# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 20:26:20 2020

@author: anama
"""

from architecture.autoencoder import autoencoderLSTM
from architecture.cnn_lstm import CNN_LSTM
from architecture.cnn_biLSTM import CNN_BiLSTM
from architecture.scb_lstm import SCB_LSTM
models = ["autoencoder LSTM", "CNN-BiLSTM", "CNN-LSTM", "SCB-LSTM", "stacked BiLSTM", "stacked LSTM"]
def training(type_model="all", timesteps=96, simulated = False, bayesian=False, save=True, validation=True):
    if type_model == "all":
        for model in models:
            print("dotrain")
            #do_train
    if type_model == "autoencoder LSTM":
         autoencoderLSTM.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save)
    elif type_model == "CNN-LSTM":
         CNN_LSTM.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, validation=False)
        
    elif type_model == "CNN-BiLSTM":
         CNN_BiLSTM.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, validation=False)
    
    elif type_model == "SCB-LSTM":
          SCB_LSTM.do_train(timesteps=timesteps, simulated = simulated, bayesian=bayesian, save=save, validation=False)