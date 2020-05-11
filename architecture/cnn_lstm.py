# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:42:20 2020

@author: anama
"""
from preprocessing.series import create_data, series_to_supervised, generate_sequences, generate_normal
from preprocessing.series import select_data
from keras.models import Sequential
from keras.layers import Dense, Reshape, RepeatVector, BatchNormalization
from keras.layers import LSTM, Activation
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
from preprocessing.series import downsample, rolling_out_cv, generate_total_sequence, select_data, csv_to_df
from keras.layers import Bidirectional
from architecture.cnn_biLSTM import CNN_BiLSTM
from architecture.scb_lstm import SCB_LSTM
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
    
    
    def decoder_Lstm(model, n_nodes, num_encdec_layers):
        model.add(LSTM(n_nodes, activation='relu', return_sequences=True))
        for i in range(0, num_encdec_layers):
             name = 'layer_lstm_decoder_{0}'.format(i+1)
             model.add(LSTM(n_nodes, activation='relu', return_sequences=True, name=name))
        model.add(LSTM(n_nodes, activation='relu', return_sequences=False))
        return model
    
    def unit_BN(model, config, is_input=False):
        toIndex = EncDec.toIndex
        n_stride = tuning.get_param(config, toIndex, "stride_size")
        n_stride = 2
        n_stride = (n_stride, )
        print("stride size", n_stride)
        n_kernel = tuning.get_param(config, toIndex, "kernel_size")
        print("n_kernel", n_kernel)
        n_filters = tuning.get_param(config, toIndex, "no_filters")
        print("n_filters", n_filters)
        n_steps = CNN_LSTM.n_steps
        n_features = CNN_LSTM.n_features
        if is_input:
            model.add(TimeDistributed(Conv1D(filters=n_filters, strides=n_stride, kernel_size=(n_kernel,), activation='relu'), input_shape=(None,n_steps,n_features)))
        else:
            model.add(TimeDistributed(Conv1D(filters=n_filters, strides=n_stride, kernel_size=(n_kernel,), activation='relu')))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        
        return model
    
    
    def multi_channel(X_train, y_train, config):
         toIndex = CNN_LSTM.toIndex
         n_steps = EncDec.n_steps
         num_pooling_layers = tuning.get_param(config, toIndex, "num_pooling_layers")
         n_stride = tuning.get_param(config, toIndex, "stride_size")
         n_stride = 2
         n_stride = (n_stride, )
      
         n_kernel = tuning.get_param(config, toIndex, "kernel_size")
         n_filters = tuning.get_param(config, toIndex, "no_filters")
      
         num_encdec_layers = tuning.get_param(config, toIndex, "num_encdec_layers")
         learning_rate = tuning.get_param(config, toIndex, "learning_rate")
         drop_rate_1 = tuning.get_param(config, toIndex, "drop_rate_1")
         n_nodes = 16
         
         n_features = X_train.shape[3]
    
         model = Sequential()
         model.add(TimeDistributed(Conv1D(filters=n_filters, strides=n_stride, kernel_size=(n_kernel,), activation='relu'), input_shape=(None,n_steps,n_features)))
         if CNN_LSTM.batch_normalization:
             
             model = CNN_LSTM.unit_BN(model, config, is_input=True)
             for i in range(0, num_encdec_layers):
                 model = CNN_LSTM.unit_BN(model, config, is_input=False)
         else:
             for i in range(0, num_encdec_layers):
                 model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=(n_kernel,), activation='relu')))
             for i in range(0, num_pooling_layers):
                 name = 'pooling_layer_{0}'.format(i+1)
                 model.add(TimeDistributed(MaxPooling1D(pool_size=2, name=name)))
             model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
         if CNN_LSTM.dropout == True:
             model.add(Dropout(drop_rate_1))
         model.add(TimeDistributed(Flatten()))  
         if CNN_LSTM.decoder == "LSTM":
             model = CNN_LSTM.decoder_Lstm(model, n_nodes, num_encdec_layers)
         elif CNN_LSTM.decoder == "BiLSTM":
             model = CNN_BiLSTM.decoder_BiLstm(model, n_nodes, num_encdec_layers)
         elif CNN_LSTM.decoder == "SCB-LSTM":
             model = SCB_LSTM.decoder_SCB_lstm(model, n_nodes, num_encdec_layers)
         if CNN_LSTM.dropout == True:
             model.add(Dropout(drop_rate_1))
         model.add(Dense(n_features))
            
   
         adam = Adam(lr=learning_rate)
         model.compile(loss='mse', optimizer=adam)
     
         print(model.summary())
         outputs = [layer.output for layer in model.layers]
         inputs = [layer.input for layer in model.layers]
         print("outputs", outputs)
         print("inputs", inputs)
       
         return model
     
   
    
    def multi_head(X_train, y_train, config): 
        toIndex = CNN_LSTM.toIndex
        n_steps = EncDec.n_steps
        num_pooling_layers = tuning.get_param(config, toIndex, "num_pooling_layers")
        stride_size = tuning.get_param(config, toIndex, "stride_size")
        n_kernel = tuning.get_param(config, toIndex, "kernel_size")
        print("config",config)
        print("toIndex", toIndex)
        print("kernel size", n_kernel)
        n_filters = tuning.get_param(config, toIndex, "no_filters")
        num_encdec_layers = tuning.get_param(config, toIndex, "num_encdec_layers")
        learning_rate = tuning.get_param(config, toIndex, "learning_rate")
        drop_rate_1 = tuning.get_param(config, toIndex, "drop_rate_1")
        n_features = X_train.shape[3]
      
        # create a channel for each time series/sensor
        in_layers, out_layers = list(), list()
     
        for n in range(n_features):
            inputs = Input(shape=(None, n_steps,1))
            print("inputs shape", inputs.shape)
            conv1 = TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))(inputs)
            conv2 = TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))(conv1)
            print("conv shape", conv2.shape)
            last_conv = conv2
            for i in range(0, num_encdec_layers):
                new_conv = TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))(last_conv)
                last_conv = new_conv
            
          
            pool1 = TimeDistributed(MaxPooling1D(pool_size=2))(last_conv)
            last_pool = pool1
            
        
            for j in range(num_pooling_layers):
                new_pool = TimeDistributed(MaxPooling1D(pool_size=2))(last_pool)
                last_pool = new_pool
       
                
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
    
              

    
    def hyperparam_opt(timesteps, n_input): 
         dimensions, default_parameters = tuning.get_param_conv_layers(timesteps, n_input)
      
         dimensions += tuning.get_param_encdec(timesteps)[0]
     
         default_parameters += tuning.get_param_encdec(timesteps)[1]
     
         EncDec.dimensions = dimensions
         EncDec.default_parameters = default_parameters
     
         for i in range(0, len(dimensions)):
             EncDec.toIndex[dimensions[i].name] = i
    
         return dimensions, default_parameters
     
   
    n_seq = 7
    n_input = n_seq * EncDec.n_steps
    dimensions, default_parameters = hyperparam_opt(EncDec.n_steps, n_input)
    config = default_parameters
    
    def __init__(self, type_model="multi-channel", model_name="CNN-LSTM", report_name=None):  
        CNN_LSTM.config = CNN_LSTM.default_parameters
        print("model_name", model_name)
        if model_name == "CNN-LSTM":
            CNN_LSTM.encoder = "CNN"
            CNN_LSTM.decoder = "LSTM"
        
        elif model_name == "CNN-BiLSTM":
                CNN_LSTM.encoder = "CNN"
                CNN_LSTM.decoder = "BiLSTM"
                
        elif model_name == "SCB-LSTM":
              CNN_LSTM.encoder = "CNN"
              CNN_LSTM.decoder = "SCB-LSTM"
        else:
            raise ValueError("No such model name")
            
        type_model_func = None
        if report_name != None:
            CNN_LSTM.report_name = report_name
            
        if type_model == "multi-head":
            CNN_LSTM.type_model_func = CNN_LSTM.multi_head
            EncDec.split = True
            if report_name == None:
                CNN_LSTM.report_name = "CNN_LSTM_multi_head_Report" + model_name
        elif type_model == "multi-channel":
            CNN_LSTM.type_model_func = CNN_LSTM.multi_channel
            if report_name == None:
                CNN_LSTM.report_name = "CNN_LSTM_multi_channel_Report" + model_name
        else:
            raise ValueError('No such architecture')
            
        
        CNN_LSTM.input_form = "4D"
        CNN_LSTM.output_form = "2D"
        CNN_LSTM.dropout = False
        CNN_LSTM.regularizer = "L1"
        CNN_LSTM.batch_normalization = True
        CNN_LSTM.use_cross_validation = True
        CNN_LSTM.no_calls_fitness = 0
        
        for i in range(0, len(CNN_LSTM.dimensions)):
             CNN_LSTM.toIndex[CNN_LSTM.dimensions[i].name] = i
        

    
              
  
    
    @use_named_args(dimensions=dimensions)
    def fitness(num_pooling_layers, stride_size, kernel_size, no_filters, num_encdec_layers, batch_size, learning_rate, drop_rate_1):  
        init = time.perf_counter()
        print("fitness>>>")
        CNN_LSTM.no_calls_fitness += 1
        print("Number of calls to fitness", CNN_LSTM.no_calls_fitness)
    
        n_steps = EncDec.n_steps
        n_features = EncDec.n_features
        
        sequence = generate_total_sequence("12", "sensortgmeasurepp",start=CNN_LSTM.stime, end=CNN_LSTM.etime)
        train_chunks, test_chunks = rolling_out_cv(sequence, 8640)
                
        normal_sequence, test_sequence = generate_sequences("12", "sensortgmeasurepp", df_to_csv=True)
        normal_sequence = generate_normal("12", df_to_csv = True)
        
        n_seq = CNN_LSTM.get_n_seq()
        print("n_seq", n_seq)
        n_input = CNN_LSTM.get_n_input()
        print("n_input", n_input)
        X_train_full, y_train_full = utils.generate_full(normal_sequence,n_steps, input_form=CNN_LSTM.input_form, output_form=CNN_LSTM.output_form, n_seq=CNN_LSTM.n_seq,n_input=CNN_LSTM.n_input, n_features=CNN_LSTM.n_features)
        
        kernel_size=5
        stride_size=2
        no_filters=15
       
        config = [num_pooling_layers, stride_size, kernel_size, no_filters, num_encdec_layers, batch_size, learning_rate,  drop_rate_1]
        
       
        print("Num pooling layers", num_pooling_layers)
        print("Stride size", stride_size)
        print("Kernel size", kernel_size)
        print("No filters", no_filters)
        print("EncDec layers", num_encdec_layers)
        print("Batch size", batch_size)
        print("Learning rate", learning_rate)
        print("Drop rate",drop_rate_1)
        input_data = utils.split_features(EncDec.n_features, X_train_full)
        model = CNN_LSTM.type_model_func(X_train_full, y_train_full, config) 
        k = len(train_chunks)
        all_losses = list()
          
        X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2 = utils.generate_sets(normal_sequence, n_steps,input_form =  CNN_LSTM.input_form, output_form = CNN_LSTM.output_form, validation=True, n_seq=CNN_LSTM.n_seq,n_input=CNN_LSTM.n_input, n_features=CNN_LSTM.n_features)
        es = EarlyStopping(monitor='val_loss', min_delta = 0.01, mode='min', verbose=1)
        input_data = list()
            
        print("CNN_LSTM TYPE", CNN_LSTM.type_model)
        hist = None
        if CNN_LSTM.type_model == "multi-channel":
              hist = model.fit(X_train, y_train, validation_data=(X_val_1, y_val_1), epochs=100, batch_size= batch_size, callbacks=[es])
        elif CNN_LSTM.type_model == "multi-head":
                input_data = utils.split_features(EncDec.n_features, X_train)
                hist = model.fit(input_data, y_train, validation_data=(X_val_1, y_val_1), epochs=200, batch_size=batch_size, callbacks=[es])
              
        loss = hist.history['loss'][-1]
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
    
       
 
 
   
 

                

            
        

    
    
    