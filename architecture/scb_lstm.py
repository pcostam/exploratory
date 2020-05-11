# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:23:32 2020

@author: anama
"""
from keras.models import Sequential
from keras.layers import Dense, Reshape, RepeatVector, Bidirectional
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from EncDec import EncDec
from skopt.space import Integer, Real

class SCB_LSTM(EncDec):
  
    def decoder_SCB_lstm(model, n_nodes, num_encdec_layers):
        num_layers = round(num_encdec_layers/2)
        print("num_layers", num_layers)
        model.add(Bidirectional(LSTM(n_nodes, activation='relu', return_sequences=True)))
        for i in range(0, num_layers):
            name = 'layer_bilstm_decoder_{0}'.format(i+1)
            model.add(Bidirectional(LSTM(n_nodes, activation='relu', return_sequences=True, name=name)))
        for i in range(0, num_layers):
             name = 'layer_lstm_decoder_{0}'.format(i+1)
             model.add(LSTM(n_nodes, activation='relu', return_sequences=True, name=name))
        model.add(LSTM(n_nodes, activation='relu', return_sequences=False))
        
        return model
        
