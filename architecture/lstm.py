# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:16:37 2019

@author: anama
"""
from keras.layers import Input, Dense , Bidirectional, LSTM
from keras.models import Model
import keras.backend as K

def q_loss(q,y_pred,y_true):
    e = (y_pred-y_true)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def model_with_quantiles(timesteps, units):
    inputs = Input(shape=(timesteps, units))  
    lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(inputs, training = True)
    lstm = Bidirectional(LSTM(16, return_sequences=False, dropout=0.3))(lstm, training = True)
    dense = Dense(50)(lstm)
    out10 = Dense(1)(dense)
    #out50 = Dense(1)(dense)
    #out90 = Dense(1)(dense)
    model = Model(inputs=inputs, outputs=out10)
    return model


def model_method(timesteps, units):
    inputs = Input(shape=(timesteps, units))  
    lstm = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(inputs, training = True)
    lstm = Bidirectional(LSTM(16, return_sequences=False, dropout=0.3))(lstm, training = True)
    dense = Dense(50)(lstm)
    out10 = Dense(1)(dense)
    out50 = Dense(1)(dense)
    out90 = Dense(1)(dense)
    model = Model(inputs=inputs, outputs=[out10,out50,out90])
    return model

    