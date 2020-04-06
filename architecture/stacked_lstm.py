# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:15:38 2020

@author: anama
"""
#See references
#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
from keras.layers import LSTM, Dense
from keras.models import Sequential
from preprocessing.series import create_dataset_as_supervised, create_dataset, generate_sequences, series_to_supervised, select_data, generate_normal
from preprocessing.series import downsample
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import skopt
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt.callbacks import DeltaYStopper
import tensorflow
from keras.backend import clear_session
from keras import regularizers
import datetime
from keras.optimizers import Adam
from architecture.evaluate import f_beta_score
import time
import pickle
import os
import utils
from Baseline import Baseline

class stacked_LSTM(Baseline):
    input_form = "3D"
    output_form = "2D"
    num_lstm_layers = 2
    learning_rate = 0.01
    batch_size = 128
    n_seq = None
    n_input = Baseline.n_steps
   
    
    def stacked_lstm_model(X, num_lstm_layers, learning_rate):
        n_steps = X.shape[1]
        n_features = X.shape[2]
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        for i in range(num_lstm_layers):
            name = 'layer_lstm_{0}'.format(i+1)
            model.add(LSTM(50, activation='relu', return_sequences=True, name=name))
        model.add(LSTM(50, activation='relu', return_sequences=False))
        model.add(Dense(n_features))
        
        adam = Adam(lr=learning_rate)
        model.compile(optimizer=adam, loss='mae',  metrics=['accuracy'])
        model.summary()
        
        return model
    
    type_model_func = stacked_lstm_model
    
    @classmethod
    def hyperparam_opt(cls):
         
      
        num_lstm_layers = Integer(low=0, high=5, name='num_lstm_layers') 
        learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
        dim_batch_size = Integer(low=64, high=128, name='batch_size')
        dimensions = [num_lstm_layers,
        learning_rate,
        dim_batch_size] 
        
        default_parameters =[2,
        0.01,
        128] 
        
        cls.dimensions = dimensions
        cls.default_parameters = default_parameters
        cls.config = cls.default_parameters
        
        for i in range(0, len(dimensions)):
             cls.toIndex[dimensions[i].name] = i
    
stacked_LSTM.hyperparam_opt()  
"""
    def test():
        stime ="01-01-2017 00:00:00"
        etime ="01-03-2017 00:00:00"
    
        normal_sequence, _ = generate_sequences("12", "sensortgmeasurepp",start=stime, end=etime, simulated=False, df_to_csv=True)
        print("test normal_sequence", normal_sequence[0].shape)
    
            
        clear_session()
        #tensorflow.reset_default_graph()
            
        #1 minute frequency size of sliding window 1440- day
        #it already seems unfeasible
        #week 10080
        #15 min frequency a day is 96
        timesteps = 3
        
        num_lstm_layers = 2
        learning_rate = 0.01
        batch_size = 128
        n_features = 1
        
        #in chunks
        X_train_full, y_train_full = utils.generate_full(normal_sequence,timesteps)
        print("X_train_full shape", X_train_full.shape)
        print("y_train_full shape", y_train_full.shape)
        model = stacked_lstm(X_train_full, num_lstm_layers, learning_rate)
        
        number_of_chunks = 0
        history = list()
        is_best_model = False
        validation = True
      
        for df_chunk in normal_sequence:
            number_of_chunks += 1
            print("number of chunks:", number_of_chunks)
            X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2  = utils.generate_sets(df_chunk, timesteps, validation=validation)  
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
            history = model.fit(X_train, y_train, epochs=20,validation_data=(X_val_1, y_val_1),  batch_size=batch_size, callbacks=[es]).history
         
        #print("X_val_1", X_val_1.shape)
        X_test = np.array([0.2, 0.5, 0.67])
        X_test = X_test.reshape((1, timesteps, n_features))
        #X_pred = model.predict(X_val_1)
        
        #print("X_pred", X_pred)
    """
          
          