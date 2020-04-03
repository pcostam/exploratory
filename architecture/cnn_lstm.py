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

#see https://machinelearningmastery.com/how-to-develop-deep-learning-models-for-univariate-time-series-forecasting/
#https://towardsdatascience.com/get-started-with-using-cnn-lstm-for-forecasting-6f0f4dde5826
#https://blog.keras.io/building-autoencoders-in-keras.html
#multi-channel
#https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/

type_model = None
type_model_func = None
config =  [1, 1, 20, 1, 2, 128, 1e-2, 0.5, 0.5]
n_steps = 96
n_seq = 10
n_input = n_seq * n_steps
n_features = 1
toIndex = dict()
n_epochs = 0
timesteps = 96
dimensions = []
default_parameters = []
fitness_func = None

def get_to_index():
    global toIndex
    return  toIndex

def get_n_steps():
    global n_steps
    return n_steps

def get_n_epochs():
    global n_epochs
    return n_epochs

def get_fitness():
    global fitness_func
    return fitness_func

def get_dimensions():
    global dimensions
    return dimensions

def get_default_parameters():
    global default_parameters
    return default_parameters

def get_n_features():
    global n_features
    return n_features 

def set_dimensions(arg):
    global dimensions
    dimensions = arg
    
def set_default_parameters(arg):
    global default_parameters
    default_parameters = arg

def set_n_features(arg):
    global n_features
    n_features = arg

def cnn_lstm(X_train, y_train, config):
    toIndex = get_to_index()
    n_steps = get_n_steps()
    num_pooling_layers = tuning.get_param(config, toIndex, "num_pooling_layers")
    n_stride = tuning.get_param(config, toIndex, "stride_size")
    n_kernel = tuning.get_param(config, toIndex, "kernel_size")
    n_filters = tuning.get_param(config, toIndex, "no_filters")
    num_encdec_layers = tuning.get_param(config, toIndex, "num_encdec_layers")
    learning_rate = tuning.get_param(config, toIndex, "learning_rate")
    drop_rate_1 = tuning.get_param(config, toIndex, "drop_rate_1")
    n_nodes = 100
    
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
    model.add(TimeDistributed(Flatten())) 
    #model.add((Dense(3)))
    #model.add(Reshape((1,2))) 
    model.add(LSTM(n_nodes, activation='relu', return_sequences=True))
    for i in range(0, num_encdec_layers):
        name = 'layer_lstm_decoder_{0}'.format(i+1)
        model.add(LSTM(n_nodes, activation='relu', return_sequences=True, name=name))
    model.add(LSTM(n_nodes, activation='relu', return_sequences=False))
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
    toIndex = get_to_index()
    n_steps = get_n_steps()
    num_pooling_layers = tuning.get_param(config, toIndex, "num_pooling_layers")
    stride_size = tuning.get_param(config, toIndex, "stride_size")
    n_kernel = tuning.get_param(config, toIndex, "kernel_size")
    n_filters = tuning.get_param(config, toIndex, "no_fitlers")
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

def hyperparam_opt(timesteps): 
    toIndex = get_to_index()
    dimensions, default_parameters = tuning.get_param_conv_layers(timesteps)
         
    dimensions += tuning.get_param_encdec(timesteps)[0]
    
    default_parameters += tuning.get_param_encdec(timesteps)[1]
    
    set_dimensions(dimensions)
    set_default_parameters(default_parameters)
    
    for i in range(0, len(dimensions)):
        toIndex[dimensions[i].name] = i
        
hyperparam_opt(timesteps)
@use_named_args(dimensions=dimensions)
def fitness(num_pooling_layers, stride_size, kernel_size, no_filters, num_encdec_layers, batch_size, learning_rate, drop_rate_1):  
    init = time.perf_counter()
    print("fitness>>>")
    n_steps = get_n_steps()
    n_features = get_n_features()
      
    _, normal_sequence, _ = generate_sequences("12", "sensortgmeasurepp", limit=True, df_to_csv=True)
    normal_sequence = generate_normal("12", limit=True, n_limit=129600, df_to_csv = True)
    
    X_train_full, y_train_full = utils.generate_full(normal_sequence, n_steps, model="CNN", n_seq=n_seq, n_input=n_input, n_features=n_features)
   
    config = [num_pooling_layers, stride_size, kernel_size, no_filters, num_encdec_layers, batch_size, learning_rate,  drop_rate_1]
    
    model = type_model_func(X_train_full, y_train_full, config) 
                      
    
    print("total number of chunks", len(normal_sequence))
    no_chunks = 0
    for df_chunk in normal_sequence:
        no_chunks += 1
        print("number of chunks:", no_chunks)
        X_train, y_train = utils.generate_sets(df_chunk, n_steps, "CNN", n_seq, n_input, n_features) 
        es = EarlyStopping(monitor='val_loss', min_delta = 0.01, mode='min', verbose=1)
        input_data = list()
        if type_model == "multi-channel":
            input_data = utils.split_features(n_features, X_train)
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

#config
#n_input: The number of lag observations to use as input to the model.
#n_filters: The number of parallel filters.
#n_kernel: The number of time steps considered in each read of the input sequence.
#n_epochs: The number of times to expose the model to the whole training dataset.
#n_batch: The number of samples within an epoch after which the weights are updated

def do_train(architecture="multi-channel", bayesian=False, simulated=False, save=True):
    global type_model, type_model_func, timesteps, toIndex
    if architecture == "multi-head":
        type_model_func = multi_head
        type_model = architecture
    elif architecture == "multi-channel":
        type_model_func = cnn_lstm
        type_model = architecture
    else:
        raise ValueError('No such architecture')
       
    set_n_features(1)
    config = get_default_parameters()
    if bayesian == True:
       
        fitness = get_fitness()
        dimensions = get_dimensions()
        default_parameters = get_default_parameters()
        
        param = tuning.do_bayesian_optimization(fitness, dimensions, default_parameters)
        config = param
        
    stime ="01-01-2017 00:00:00"
    etime ="01-03-2017 00:00:00"
    
    normal_sequence, _ = generate_sequences("12", "sensortgmeasurepp",start=stime, end=etime, simulated=simulated, df_to_csv=True)
    normal_sequence_2, _ = generate_sequences("11", "sensortgmeasurepp",start=stime, end=etime, simulated=simulated, df_to_csv=True)
   
    list_result = utils.concatenate_features(normal_sequence, normal_sequence_2)
    print("list_result", list_result)
    n_features = 2
    normal_sequence = list_result
    
    print("to index", toIndex["batch_size"] )
    print("dimensions", len(config))
    batch_size = tuning.get_param(config, toIndex, "batch_size")
      
    X_train_full, y_train_full = utils.generate_full(normal_sequence, n_steps, model="CNN", n_input=n_input, n_features=n_features)
    print("y_train_full shape", y_train_full.shape)
    print("x_train_full shape", X_train_full.shape)
    
    
    model = type_model_func(X_train_full, y_train_full, config)
    number_of_chunks = 0
    for df_chunk in normal_sequence:
        #if is_best_model:
        #    model = load_model("best_autoencoderLSTM.h5")
        number_of_chunks += 1
        print("number of chunks:", number_of_chunks)
        X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2 = utils.generate_sets(df_chunk, timesteps, validation=False, type_input="CNN",n_seq = n_seq,n_input=n_input,n_features=n_features)  
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        mc = ModelCheckpoint('best_autoencoderLSTM.h5', monitor='val_loss', mode='min', save_best_only=True)
        history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size, callbacks=[es, mc]).history
    
        
    
    return True
    
    


    
    
    