# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:20:53 2020

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

class CNN_BiLSTM(EncDec):
  
    
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
    
    def model(X_train, y_train, config):
        toIndex = EncDec.toIndex
      
        n_steps = EncDec.n_steps
        #num_pooling_layers = tuning.get_param(config, toIndex, "num_pooling_layers")
        #n_stride = tuning.get_param(config, toIndex, "stride_size")
        n_kernel = tuning.get_param(config, toIndex, "kernel_size")
        n_filters = tuning.get_param(config, toIndex, "no_filters")
        num_encdec_layers = tuning.get_param(config, toIndex, "num_encdec_layers")
        learning_rate = tuning.get_param(config, toIndex, "learning_rate")
        drop_rate_1 = tuning.get_param(config, toIndex, "drop_rate_1")
        n_nodes = 50
        n_features = X_train.shape[3]
        print("n_features", n_features)
    
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'), input_shape=(None,n_steps,n_features)))
        model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu')))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten())) 
        #model.add((Dense(3)))
        #model.add(Reshape((1,2))) 
        model.add(Bidirectional(LSTM(n_nodes, activation='relu', return_sequences=True)))
        model.add(Bidirectional(LSTM(n_nodes, activation='relu', return_sequences=False)))
        """
        for i in range(num_encdec_layers):
            name = 'layer_lstm_decoder_{0}'.format(i+1)
        """ 
        model.add(Dropout(0.2))
        model.add(Dense(n_features))
     
        
       
        #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
          
        adam = Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer=adam)
      
        print(model.summary())
 
      
        return model
    
    type_model_func = model
    
    
    def hyperparam_opt(timesteps, n_input): 
         dimensions, default_parameters = tuning.get_param_conv_layers(timesteps, n_input)
      
         dimensions += tuning.get_param_encdec(timesteps)[0]
     
         default_parameters += tuning.get_param_encdec(timesteps)[1]
     
         dimensions = dimensions
         default_parameters = default_parameters
     
         for i in range(0, len(dimensions)):
             EncDec.toIndex[dimensions[i].name] = i
             
         return dimensions, default_parameters
     
   
    
    def __init__(self, report_name=None):
          CNN_BiLSTM.config =  [1, 1, 20, 1, 2, 128, 1e-2, 0.5, 0.5]
          CNN_BiLSTM.input_form = "4D"
          CNN_BiLSTM.output_form = "2D"
          if report_name != None:
            CNN_BiLSTM.report_name = report_name
          else:
            CNN_BiLSTM.report_name = "Cnn_bilstm_report"
            
         
            
    
    n_seq = 7
    n_input = n_seq * EncDec.n_steps
    dimensions, default_parameters = hyperparam_opt(EncDec.n_steps, n_input)
        
    
    @use_named_args(dimensions=EncDec.dimensions)
    def fitness(num_pooling_layers, stride_size, kernel_size, no_filters, num_encdec_layers, batch_size, learning_rate, drop_rate_1):  
        init = time.perf_counter()
        print("fitness>>>")
        n_steps = EncDec.n_steps
        n_features = EncDec.n_features
        
        _, normal_sequence, _ = generate_sequences("12", "sensortgmeasurepp", limit=True, df_to_csv=True)
        normal_sequence = generate_normal("12", limit=True, n_limit=129600, df_to_csv = True)
        
        n_seq = CNN_BiLSTM.get_n_seq()
        n_input = CNN_BiLSTM.get_n_input()
        X_train_full, y_train_full = utils.generate_full(normal_sequence, n_steps, model="CNN", n_seq=n_seq, n_input=n_input, n_features=n_features)
       
        config = [num_pooling_layers, stride_size, kernel_size, no_filters, num_encdec_layers, batch_size, learning_rate,  drop_rate_1]
    
        model = CNN_BiLSTM.type_model_func(X_train_full, y_train_full, config) 
                      
    
        print("total number of chunks", len(normal_sequence))
        no_chunks = 0
        for df_chunk in normal_sequence:
            no_chunks += 1
            print("number of chunks:", no_chunks)
            X_train, y_train = utils.generate_sets(df_chunk, n_steps,input_form=CNN_BiLSTM.get_input_form(), output_form=CNN_BiLSTM.get_output_form(),n_seq=n_seq,n_input=n_input, n_features=n_features) 
            es = EarlyStopping(monitor='val_loss', min_delta = 0.01, mode='min', verbose=1)
            input_data = list()
            if CNN_BiLSTM.type_model == "multi-channel":
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