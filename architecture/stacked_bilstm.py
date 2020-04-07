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
    toIndex = dict()
    input_form = "3D"
    output_form = "2D"
    config = []
    n_seq = None
    n_input = Baseline.n_steps
    config = []
    h5_file_name = "stackedBiLSTM"
        
    #See references
    #https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
    def stacked_bilstm_model(X, y, config):
        toIndex = stacked_BiLSTM.toIndex 
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
    
    type_model_func = stacked_bilstm_model
    
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
    
stacked_BiLSTM.hyperparam_opt()  

