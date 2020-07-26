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
from EncDec import EncDec
import tuning
from preprocessing.splits import rolling_out_cv

class stacked_LSTM(EncDec):
  
    def stacked_lstm_model(X, y, config):
        toIndex = stacked_LSTM.toIndex 
        print("To Index", toIndex)
        num_lstm_layers = tuning.get_param(config, toIndex, "num_lstm_layers")
        learning_rate = tuning.get_param(config, toIndex, "learning_rate")
        no_nodes = tuning.get_param(config, toIndex, "num_nodes")
        
        n_steps = X.shape[1]
        n_features = X.shape[2]
        model = Sequential()
        model.add(LSTM(no_nodes, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        for i in range(num_lstm_layers):
            name = 'layer_lstm_{0}'.format(i+1)
            model.add(LSTM(no_nodes, activation='relu', return_sequences=True, name=name))
        model.add(LSTM(no_nodes, activation='relu', return_sequences=False))
        model.add(Dense(n_features))
        
        adam = Adam(lr=learning_rate)
        model.compile(optimizer=adam, loss='mae')
        model.summary()
        
        return model
    
  
 
    def hyperparam_opt():
        num_lstm_layers = Integer(low=0, high=5, name='num_lstm_layers') 
        learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
        dim_batch_size = Integer(low=64, high=128, name='batch_size')
        dim_num_nodes = Integer(low=16, high=128, name='num_nodes') 
        dimensions = [num_lstm_layers,
        learning_rate,
        dim_batch_size,
        dim_num_nodes] 
        
        default_parameters =[2,
        0.01,
        128,
        16] 
        
        return dimensions, default_parameters
    
    dimensions, default_parameters = hyperparam_opt()  
    config = default_parameters
    
    def __init__(self, report_name=None):
          stacked_LSTM.model_name = "stackedLSTM"
          stacked_LSTM.toIndex = dict()
          stacked_LSTM.input_form = "3D"
          stacked_LSTM.output_form = "2D"
          stacked_LSTM.num_lstm_layers = 2
          stacked_LSTM.learning_rate = 0.01
          stacked_LSTM.batch_size = 128
          stacked_LSTM.h5_file_name = "stackedLSTM"
          stacked_LSTM.type_model_func = stacked_LSTM.stacked_lstm_model
          stacked_LSTM.no_calls_fitness = 0
          
          if report_name == None:
             stacked_LSTM.report_name = "stacked_lstm_report"
          else:
             stacked_LSTM.report_name = report_name
          print("stacked_Lstm dim", stacked_LSTM.dimensions)
          for i in range(0, len(stacked_LSTM.dimensions)):
             stacked_LSTM.toIndex[stacked_LSTM.dimensions[i].name] = i
          print("stacked LSTM", stacked_LSTM.toIndex)
        

        
      
    @use_named_args(dimensions=dimensions)
    def fitness(num_lstm_layers, learning_rate, batch_size, num_nodes):  
        init = time.perf_counter()
        stacked_LSTM.parameters = EncDec.parameters
        print("fitness>>>")
        stacked_LSTM.no_calls_fitness += 1
        print("Number of calls to fitness", stacked_LSTM.no_calls_fitness)    
        n_steps = EncDec.parameters.get_n_steps()
        n_features = EncDec.parameters.get_n_features() 
        n_seq = EncDec.parameters.get_n_seq()
        n_input = EncDec.parameters.get_n_input()
        normal_sequence = EncDec.normal_sequence
        normal_sequence = utils.fit_transform_data(normal_sequence)
        all_losses = list()
        #n_train 3 meses
        #3*31*24*6 = 13392
        train_chunks, test_chunks = rolling_out_cv(normal_sequence, 13392, test_split=0.2, gap=0, blocked=True)
        print("NUMBER OF PARTITIONS BAYESIAN", len(train_chunks))
        #test_split is validation
        for i in range(0, len(train_chunks)):
            normal_sequence = train_chunks[i]
            print("fold shape")
            print("normal sequence", normal_sequence.shape)
            print("output form",stacked_LSTM.output_form)
            print("input form", stacked_LSTM.input_form)
            X_train_full, y_train_full = utils.generate_full(normal_sequence, n_steps, input_form = stacked_LSTM.input_form, output_form = stacked_LSTM.output_form, n_seq=n_seq,n_input=n_input, n_features=n_features)
            config = [num_lstm_layers, learning_rate, batch_size, num_nodes]
            model = stacked_LSTM.type_model_func(X_train_full, y_train_full, config) 
            X_train, y_train, _, _, _, _ = utils.generate_sets(normal_sequence, n_steps,input_form =  stacked_LSTM.input_form, output_form = stacked_LSTM.output_form, validation=False, n_seq=stacked_LSTM.parameters.get_n_seq(),n_input=stacked_LSTM.parameters.get_n_input(), n_features=stacked_LSTM.parameters.get_n_features())
            X_val, y_val, _, _, _, _ = utils.generate_sets(test_chunks[i], n_steps,input_form =  stacked_LSTM.input_form, output_form = stacked_LSTM.output_form, validation=False, n_seq=stacked_LSTM.parameters.get_n_seq(),n_input=stacked_LSTM.parameters.get_n_input(), n_features=stacked_LSTM.parameters.get_n_features())
            es = EarlyStopping(monitor='val_loss', min_delta = 0.01, mode='min', verbose=1)
            hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size= batch_size, callbacks=[es])
            
            loss = hist.history['val_loss'][-1]
            all_losses.append(loss)
            
         
        mean_loss = np.mean(np.array(all_losses))
        loss = mean_loss
        del model
        
        clear_session()
        tensorflow.compat.v1.reset_default_graph()
    
        end = time.perf_counter()
        diff = end - init
        
        return loss, diff
    
    fitness_func = fitness
    
        
        

          
          