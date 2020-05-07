# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:18:39 2020

@author: anama
"""

from keras.layers import LSTM, Dense, Bidirectional
from keras.models import Sequential
from Baseline import Baseline
from keras.optimizers import Adam
import tuning
from skopt.space import Integer, Real

class stacked_BiLSTM(Baseline):        
    #See references
    #https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    def stacked_bilstm_model(X, y, config):
        toIndex = Baseline.toIndex 
        num_lstm_layers = tuning.get_param(config, toIndex, "num_lstm_layers")
        learning_rate = tuning.get_param(config, toIndex, "learning_rate")
        
        n_steps = X.shape[1]
        n_features = X.shape[2]
        
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features))))
        
        for i in range(num_lstm_layers):
            name = 'layer_lstm_encoder_{0}'.format(i+1)
            model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True, name=name)))
        model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=False)))
        model.add(Dense(1))
        
        adam = Adam(lr=learning_rate)
        model.compile(optimizer=adam, loss='mae',  metrics=['accuracy'])
        
        
        return model
    
  
    
    def hyperparam_opt():
        num_lstm_layers = Integer(low=0, high=5, name='num_lstm_layers') 
        learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
        dim_batch_size = Integer(low=64, high=128, name='batch_size')
        dimensions = [num_lstm_layers,
        learning_rate,
        dim_batch_size] 
        
        default_parameters =[2,
        0.01,
        128] 
        
      
        for i in range(0, len(dimensions)):
             Baseline.toIndex[dimensions[i].name] = i
             
        return dimensions, default_parameters
    
    dimensions, default_parameters = hyperparam_opt() 
    config = default_parameters 
    
    def __init__(self, report_name=None):     
        stacked_BiLSTM.input_form = "3D"
        stacked_BiLSTM.output_form = "2D"
        stacked_BiLSTM.n_seq = None
        stacked_BiLSTM.n_input = Baseline.n_steps  
        stacked_BiLSTM.h5_file_name = "stackedBiLSTM"
        stacked_BiLSTM.type_model_func = stacked_BiLSTM.stacked_bilstm_model
        
        if report_name == None:
            stacked_BiLSTM.report_name = "stacked_bilstm_report"
        else:
            stacked_BiLSTM.report_name = report_name
        


