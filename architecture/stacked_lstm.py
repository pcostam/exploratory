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
import tuning

class stacked_LSTM(Baseline):
  
    def stacked_lstm_model(X, y, config):
        toIndex = Baseline.toIndex 
        print("To Index", toIndex)
        num_lstm_layers = tuning.get_param(config, toIndex, "num_lstm_layers")
        learning_rate = tuning.get_param(config, toIndex, "learning_rate")
        
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
          stacked_LSTM.input_form = "3D"
          stacked_LSTM.output_form = "2D"
          stacked_LSTM.num_lstm_layers = 2
          stacked_LSTM.learning_rate = 0.01
          stacked_LSTM.batch_size = 128
          stacked_LSTM.n_seq = None
          stacked_LSTM.n_input = Baseline.n_steps
          stacked_LSTM.h5_file_name = "stackedLSTM"
          stacked_LSTM.type_model_func = stacked_LSTM.stacked_lstm_model
          
          if report_name == None:
             stacked_LSTM.report_name = "stacked_lstm_report"
          else:
             stacked_LSTM.report_name = report_name
        
        
        

          
          