# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:42:20 2020

@author: anama
"""
from preprocessing.series import create_data, series_to_supervised, generate_sequences, generate_normal
from preprocessing.series import select_data
from keras.models import Sequential
from keras.layers import Dense, Reshape, RepeatVector
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.layers.merge import concatenate
from keras.layers import Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import datetime
import skopt
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt.callbacks import DeltaYStopper
import tuning
from skopt.utils import use_named_args
from skopt.callbacks import DeltaYStopper
import utils
import time
import tensorflow
from keras.backend import clear_session
from keras.optimizers import Adam
from EncDec import EncDec
#see https://machinelearningmastery.com/how-to-develop-deep-learning-models-for-univariate-time-series-forecasting/
#https://towardsdatascience.com/get-started-with-using-cnn-lstm-for-forecasting-6f0f4dde5826
#https://blog.keras.io/building-autoencoders-in-keras.html
#multi-channel
#https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/

#config
#n_input: The number of lag observations to use as input to the model.
#n_filters: The number of parallel filters.
#n_kernel: The number of time steps considered in each read of the input sequence.
#n_epochs: The number of times to expose the model to the whole training dataset.
#n_batch: The number of samples within an epoch after which the weights are updated

  

class CNN_LSTM(EncDec):
    report_name = "CNN_LSTMReport"
    n_seq = 7
    n_input = n_seq * EncDec.n_steps
    input_form = "4D"
    output_form = "2D"
    dropout = False
    regularizer = "L1"
    batch_normalization = False
    
    @classmethod
    def get_n_seq(cls):
        return cls.n_seq
    @classmethod
    def get_n_input(cls):
        return cls.n_input
    @classmethod
    def get_input_form(cls):
        return cls.input_form
    @classmethod
    def get_output_form(cls):
        return cls.output_form
    
    
    def multi_channel(X_train, y_train, config):
         toIndex = EncDec.toIndex
         n_steps = EncDec.n_steps
         num_pooling_layers = tuning.get_param(config, toIndex, "num_pooling_layers")
         n_stride = tuning.get_param(config, toIndex, "stride_size")
         print("stride size", n_stride)
         n_kernel = tuning.get_param(config, toIndex, "kernel_size")
         print("n_kernel", n_kernel)
         n_filters = tuning.get_param(config, toIndex, "no_filters")
         print("n_filters", n_filters)
         num_encdec_layers = tuning.get_param(config, toIndex, "num_encdec_layers")
         learning_rate = tuning.get_param(config, toIndex, "learning_rate")
         drop_rate_1 = tuning.get_param(config, toIndex, "drop_rate_1")
         n_nodes = 16
         
         n_features = X_train.shape[3]
         print("n_features", n_features)
         
         model = Sequential()
         model.add(TimeDistributed(Conv1D(filters=n_filters, strides=n_stride, kernel_size=n_kernel, activation='relu'), input_shape=(None,n_steps,n_features)))
         for i in range(0, num_encdec_layers):
             model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu')))
         for i in range(0, num_pooling_layers):
             name = 'pooling_layer_{0}'.format(i+1)
             model.add(TimeDistributed(MaxPooling1D(pool_size=2, name=name)))
         model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
         if CNN_LSTM.dropout == True:
             model.add(Dropout(drop_rate_1))
         model.add(TimeDistributed(Flatten())) 
         #model.add((Dense(3)))
         #model.add(Reshape((1,2))) 
         model.add(LSTM(n_nodes, activation='relu', return_sequences=True))
         for i in range(0, num_encdec_layers):
             name = 'layer_lstm_decoder_{0}'.format(i+1)
             model.add(LSTM(n_nodes, activation='relu', return_sequences=True, name=name))
         model.add(LSTM(n_nodes, activation='relu', return_sequences=False))
         if CNN_LSTM.dropout == True:
             model.add(Dropout(drop_rate_1))
         model.add(Dense(n_features))
            
               
         #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
       
         adam = Adam(lr=learning_rate)
         model.compile(loss='mse', optimizer=adam)
     
         print(model.summary())
         outputs = [layer.output for layer in model.layers]
         inputs = [layer.input for layer in model.layers]
         print("outputs", outputs)
         print("inputs", inputs)
       
         return model
     
    def multi_head(X_train, y_train, config): 
        toIndex = EncDec.toIndex
        n_steps = EncDec.n_steps
        num_pooling_layers = tuning.get_param(config, toIndex, "num_pooling_layers")
        stride_size = tuning.get_param(config, toIndex, "stride_size")
        n_kernel = tuning.get_param(config, toIndex, "kernel_size")
        n_filters = tuning.get_param(config, toIndex, "no_filters")
        num_encdec_layers = tuning.get_param(config, toIndex, "num_encdec_layers")
        learning_rate = tuning.get_param(config, toIndex, "learning_rate")
        drop_rate_1 = tuning.get_param(config, toIndex, "drop_rate_1")
                       
        print("X_train shape", X_train.shape)
        print("y train shape", y_train.shape)
        n_features = X_train.shape[3]
       
        print("no features", n_features)
        # create a channel for each time series/sensor
        in_layers, out_layers = list(), list()
     
        for i in range(n_features):
            inputs = Input(shape=(None, n_steps,1))
            print("inputs shape", inputs.shape)
            conv1 = TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))(inputs)
            conv2 = TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))(conv1)
            print("conv shape", conv2.shape)
            list_conv = list()
            for i in range(0, num_encdec_layers):
                list_conv.append(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu')))
          
            last_conv = None
            group_conv = list()
            for i in range(0, len(list_conv)):
                try:
                    back_conv = list_conv[i-1]
                    new_conv = list_conv[i]
                    group_conv[i] = new_conv(back_conv)
                except IndexError:
                    back_conv = conv2
                    new_conv = list_conv[i]
                    group_conv[i] = new_conv(back_conv)
                
            last_conv = group_conv[len(group_conv)-1]
            list_pool = list()
            group_pool = list()
            pool1 = TimeDistributed(MaxPooling1D(pool_size=2))(last_conv)
            print("pool1 shape", pool1.shape)
            for i in range(0, num_pooling_layers):
                name = 'pooling_layer_{0}'.format(i+1)
                list_pool.append(TimeDistributed(MaxPooling1D(pool_size=2)), name=name)
                
            for i in range(0, len(list_pool)):
                try:
                    back_pool = list_pool[i-1]
                    new_pool = list_pool[i]
                    group_pool[i] = new_pool(back_pool)
                except IndexError:
                    pass
                
            
            last_pool = group_pool[len(list_pool)-1]
            flat = TimeDistributed(Flatten())(last_pool)
            print("flat config", flat.shape)
            
            # store layers
            in_layers.append(inputs)
            out_layers.append(flat)
            
        # merge heads
        merged = concatenate(out_layers)
        recurrent_1 = LSTM(100, activation='relu', return_sequences=True)(merged)
        recurrent_2 = LSTM(100, activation='relu', return_sequences=False)(recurrent_1)
        dropout_1 = Dropout(drop_rate_1)(recurrent_2)
        outputs= Dense(n_features)(dropout_1)
        
        model = Model(inputs=in_layers, outputs=outputs)
    	# compile model
        adam = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam)
        
        print("end compile")
        return model
    
              
    type_model = "multi-channel"
    type_model_func = None


    if type_model == "multi-head":
            type_model_func = multi_head
    elif type_model == "multi-channel":
            type_model_func = multi_channel
    else:
        raise ValueError('No such architecture')
    
    
    def hyperparam_opt(timesteps): 
         dimensions, default_parameters = tuning.get_param_conv_layers(timesteps)
      
         dimensions += tuning.get_param_encdec(timesteps)[0]
     
         default_parameters += tuning.get_param_encdec(timesteps)[1]
     
         EncDec.dimensions = dimensions
         EncDec.default_parameters = default_parameters
     
         for i in range(0, len(dimensions)):
             EncDec.toIndex[dimensions[i].name] = i
             
         return dimensions, default_parameters
     
    dimensions, default_parameters = hyperparam_opt(EncDec.n_steps)
    config = default_parameters
    @use_named_args(dimensions=EncDec.dimensions)
    def fitness(num_pooling_layers, stride_size, kernel_size, no_filters, num_encdec_layers, batch_size, learning_rate, drop_rate_1):  
        init = time.perf_counter()
        print("fitness>>>")
        n_steps = EncDec.n_steps
        n_features = EncDec.n_features
        
        _, normal_sequence, _ = generate_sequences("12", "sensortgmeasurepp", limit=True, df_to_csv=True)
        normal_sequence = generate_normal("12", limit=True, n_limit=129600, df_to_csv = True)
        
        n_seq = CNN_LSTM.get_n_seq()
        n_input = CNN_LSTM.get_n_input()
        X_train_full, y_train_full = utils.generate_full(normal_sequence, n_steps, model="CNN", n_seq=n_seq, n_input=n_input, n_features=n_features)
       
        config = [num_pooling_layers, stride_size, kernel_size, no_filters, num_encdec_layers, batch_size, learning_rate,  drop_rate_1]
    
        model = CNN_LSTM.type_model_func(X_train_full, y_train_full, config) 
                      
    
        print("total number of chunks", len(normal_sequence))
        no_chunks = 0
        for df_chunk in normal_sequence:
            no_chunks += 1
            print("number of chunks:", no_chunks)
            X_train, y_train = utils.generate_sets(df_chunk, n_steps,input_form=CNN_LSTM.get_input_form(), output_form=CNN_LSTM.get_output_form(),n_seq=n_seq,n_input=n_input, n_features=n_features) 
            es = EarlyStopping(monitor='val_loss', min_delta = 0.01, mode='min', verbose=1)
            input_data = list()
            if CNN_LSTM.type_model == "multi-channel":
                input_data = utils.split_features(EncDec.n_features, X_train)
                hist = model.fit(input_data, y_train, epochs=100, batch_size= batch_size, callbacks=[es])
            else:
                hist = model.fit(X_train, y_train, epochs=100, batch_size= batch_size, callbacks=[es])
    
        loss = hist.history['loss'][-1]
    
        del model
    
        clear_session()
        tensorflow.compat.v1.reset_default_graph()
    
        end = time.perf_counter()
        diff = end - init
    
        return loss, diff
    
      
    fitness_func = fitness
     
           
    
   
       
 
 
   
 

                

            
        

    
    
    