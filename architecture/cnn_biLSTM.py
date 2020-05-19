# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:20:53 2020

@author: anama
"""

from keras.layers import Bidirectional
from keras.layers import LSTM
from EncDec import EncDec
class CNN_BiLSTM(EncDec):
  
  def decoder_BiLstm(model, n_nodes, num_encdec_layers):
        model.add(Bidirectional(LSTM(n_nodes, activation='relu', return_sequences=True)))
        for i in range(0, num_encdec_layers):
            name = 'layer_bilstm_decoder_{0}'.format(i+1)
            model.add(Bidirectional(LSTM(n_nodes, activation='relu', return_sequences=True, name=name)))
        model.add(Bidirectional(LSTM(n_nodes, activation='relu', return_sequences=False)))
        return model 