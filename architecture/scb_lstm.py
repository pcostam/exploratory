# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:23:32 2020

@author: anama
"""
from preprocessing.series import create_data, series_to_supervised, generate_sequences, generate_normal
from preprocessing.series import select_data
from keras.models import Sequential
from keras.layers import Dense, Reshape, RepeatVector, Bidirectional
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from skopt.utils import use_named_args
import tuning
import utils
import time
import tensorflow
from keras.backend import clear_session
from keras.optimizers import Adam
from EncDec import EncDec
from keras.callbacks import EarlyStopping
from skopt.space import Integer, Real

import tuning
class SCB_LSTM(EncDec):
    toIndex = dict()
    default_parameters =[2,
        2,
        2,
        2,
        2,
        0.1,
        20,
        128]
    n_seq = 2
    n_input = n_seq * EncDec.n_steps
    input_form = "4D"
    output_form = "2D"
    
    def model(X_train, y_train, config):
        
        toIndex = SCB_LSTM.toIndex 
        n_steps = SCB_LSTM.n_steps
        n_features = X_train.shape[3]
            
        num_cnn_layers = tuning.get_param(config, toIndex, "num_cnn_layers")
        num_filters_encoder = tuning.get_param(config, toIndex, "num_filters_encoder")
        num_filters_decoder = tuning.get_param(config, toIndex, "num_filters_decoder")
        num_bi_lstm_layers =  tuning.get_param(config, toIndex, "num_bi_lstm_layers")
        num_lstm_layers = tuning.get_param(config, toIndex, "num_lstm_layers")
        learning_rate = tuning.get_param(config, toIndex, "learning_rate")
        n_kernel = tuning.get_param(config, toIndex, "kernel_size") 
       
        n_nodes = 50
        n_features = X_train.shape[3]
                
        print("n_features", n_features)
        print("n_kernel", n_kernel)

        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=num_filters_encoder, kernel_size=n_kernel, activation='relu'), input_shape=(None,n_steps,n_features)))
        model.add(TimeDistributed(Conv1D(filters=num_filters_decoder, kernel_size=n_kernel, activation='relu')))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten())) 
        #model.add((Dense(3)))
        #model.add(Reshape((1,2))) 
        model.add(Bidirectional(LSTM(n_nodes, activation='relu', return_sequences=True)))
        """
        for i in range(num_bi_lstm_layers):
            name = 'layer_bi_lstm{0}'.format(i+1)
            model.add(Bidirectional(LSTM(n_nodes, activation='relu', return_sequences=False, name=name)))
        """
        model.add(LSTM(n_nodes, activation='relu', return_sequences=False))
        """
        for i in range(num_lstm_layers):
            name = 'layer_lstm{0}'.format(i+1)
        """ 
        
            
        model.add(Dropout(0.2))
        model.add(Dense(n_features))
        
        adam = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam)
        
        return model
    
    type_model_func = model
     
    @classmethod
    def hyperparam_opt(cls):
        num_cnn_layers = Integer(low=0, high=20, name='num_cnn_layers')
        num_filters_encoder= Integer(low=0, high=5, name='num_filters_encoder')
        num_filters_decoder = Integer(low=0, high=5, name='num_filters_decoder')
        num_bi_lstm_layers = Integer(low=0, high=20, name='num_bi_lstm_layers') 
        num_lstm_layers = Integer(low=0, high=20, name='num_lstm_layers') 
        learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
        dim_kernel = Integer(low=0, high=5, name='kernel_size')
        dim_batch_size = Integer(low=64, high=128, name='batch_size')
        dimensions = [num_cnn_layers,
        num_filters_encoder,
        num_filters_decoder,
        num_bi_lstm_layers,
        num_lstm_layers,
        learning_rate,
        dim_kernel,
        dim_batch_size] 
        
        default_parameters =[2,
        2,
        2,
        2,
        2,
        0.1,
        20,
        128] 
        
        cls.dimensions = dimensions
        cls.default_parameters = default_parameters
        cls.config = cls.default_parameters
        
        for i in range(0, len(dimensions)):
             cls.toIndex[dimensions[i].name] = i
    
          
SCB_LSTM.hyperparam_opt()
        

    