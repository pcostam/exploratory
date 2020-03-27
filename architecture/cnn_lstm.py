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


#see https://machinelearningmastery.com/how-to-develop-deep-learning-models-for-univariate-time-series-forecasting/
#https://towardsdatascience.com/get-started-with-using-cnn-lstm-for-forecasting-6f0f4dde5826
#https://blog.keras.io/building-autoencoders-in-keras.html
#multi-channel
#https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/


def generate_sets(raw, n_seq, n_input, timesteps, n_features):
    y_train = preprocess_y_train(raw, n_input, n_features)
    print("y_train shape", y_train.shape)
   
    X_train = preprocess_train(raw, n_seq, n_input, timesteps, n_features)
    
    print("type xtrain", type(X_train))
    print("type ytrain", type(y_train))
    
    return X_train, y_train

def preprocess_y_train(raw, n_input, n_features):
    raw = raw.drop(['date'], axis=1)
    data = series_to_supervised(raw, n_in=n_input)
    print("data to supervised", data)
    data = data.values
    print("data", type(data))
    print("data", data)
    
    if n_features == 1:
        y_train = [data[:, -1]]
    else:
        y_train = data[:, :n_features]
    print("y_train_full", y_train)
    
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_train)
    
    y_train =  np.squeeze(y_train)
    
    return y_train

def preprocess_train(raw, n_seq, n_input, timesteps, n_features):
    raw = raw.drop(['date'], axis = 1)
    print("raw", raw)
    print("raw columns", raw.columns)
    print("raw index", raw.index)
 
    data = series_to_supervised(raw, n_in=n_input)
    print("data shape 2", data.shape)
    data = np.array(data.iloc[:, :n_input])
    print("data shape 3", data.shape)
    #normalize data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    print("data", data)
    
    
    #for CNN there is the need to reshape de 2-d np array to a 4-d np array [samples, timesteps, features]
    #[batch_size, height, width, depth]
    #[samples, subsequences, timesteps, features]
    print("samples", data.shape[0])
    print("subsequences", n_seq)
    print("timesteps", timesteps)
    print("features", n_features)
    
    rows = data.shape[0] * data.shape[1]
    new_n_seq = round(rows/(data.shape[0]*timesteps*n_features))

    n_seq = new_n_seq
    data = np.reshape(data, (data.shape[0], n_seq, timesteps, n_features))
    
    return data



def generate_full_y_train(normal_sequence, n_input, timesteps, n_features):
    y_train_full = list()
    size = len(normal_sequence)
    if size  > 1:
        y_train_full = pd.concat(normal_sequence)
    else:
        print("normal_sequence", normal_sequence)
        print("size", len(normal_sequence))
        print(normal_sequence[0])
        y_train_full = normal_sequence[0]
    
    print(type(y_train_full))
    stime ="01-01-2017 00:00:00"
    etime ="01-03-2017 00:00:00"
        
    frmt = '%d-%m-%Y %H:%M:%S'
    min_date = datetime.datetime.strptime(stime, frmt)
    max_date = datetime.datetime.strptime(etime, frmt)
    print("min date", type(min_date))
    print("type", y_train_full.dtypes)
    
    y_train_full = select_data(y_train_full, min_date, max_date)
    y_train_full = y_train_full.drop(['date'], axis=1)
    data = series_to_supervised(y_train_full, n_in=n_input)
    print("data to supervised", data)
    data = data.values
    print("data", type(data))
    print("data", data)
    
    if n_features == 1:
        y_train_full = [data[:, -1]]
    else:
        y_train_full = data[:, :n_features]
    print("y_train_full", y_train_full)
    
    
    
    scaler = MinMaxScaler()
    y_train_full = scaler.fit_transform(y_train_full)
    
    y_train_full =  np.squeeze(y_train_full)
    
    return y_train_full

#copiar isto para pai
def generate_full_X_train(normal_sequence, n_seq, n_input, timesteps, n_features):
    X_train_full = list()
    size = len(normal_sequence)
    if size  > 1:
        X_train_full = pd.concat(normal_sequence)
    else:
        print("normal_sequence", normal_sequence)
        print("size", len(normal_sequence))
        print(normal_sequence[0])
        X_train_full = normal_sequence[0]
        
    print("after concantening pieces", X_train_full)
    print(type(X_train_full))
    stime ="01-01-2017 00:00:00"
    etime ="01-03-2017 00:00:00"

    frmt = '%d-%m-%Y %H:%M:%S'
    min_date = datetime.datetime.strptime(stime, frmt)
    max_date = datetime.datetime.strptime(etime, frmt)
    print("min date", type(min_date))
    print("type", X_train_full.dtypes)
    
    X_train_full = select_data(X_train_full, min_date, max_date)
    print("X_train_full after select data", X_train_full)
    X_train_full = preprocess_train(X_train_full, n_seq, n_input, timesteps, n_features)
    print("X_train_full shape>>>", X_train_full.shape)
    return X_train_full


def cnn_lstm(X_train, y_train, n_epochs, n_batch, n_kernel, n_filters, n_nodes, n_steps):
    n_features = X_train.shape[3]
    print("n_features", n_features)
    size = n_steps*n_filters
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'), input_shape=(None,n_steps,n_features)))
    model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten())) 
    #model.add((Dense(3)))
    #model.add(Reshape((1,2))) 
    model.add(LSTM(n_nodes, activation='relu', return_sequences=True))
    model.add(LSTM(n_nodes, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(n_features))
 
    
   
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    model.compile(loss='mse', optimizer='adam')
  
    print(model.summary())
    outputs = [layer.output for layer in model.layers]
    inputs = [layer.input for layer in model.layers]
    print("outputs", outputs)
    print("inputs", inputs)
    
  
    return model

def multi_head(X_train, y_train, n_epochs, n_batch, n_kernel, n_filters, n_nodes, n_steps):
    print("X_train shape", X_train.shape)
    print("y train shape", y_train.shape)
    n_features = X_train.shape[3]
    input_data = list()
    
    for i in range(n_features):
        aux = X_train[:, :, :, i]
        print("aux shape", aux.shape)
        reshaped = aux.reshape((aux.shape[0], aux.shape[1], aux.shape[2], 1))
        input_data.append(reshaped)
     
        print("reshaped", reshaped.shape)
      
   
    print("no features", n_features)
    # create a channel for each time series/sensor
    in_layers, out_layers = list(), list()
 
    for i in range(n_features):
        inputs = Input(shape=(None, n_steps,1))
        print("inputs shape", inputs.shape)
        conv1 = TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))(inputs)
        conv2 = TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'))(conv1)
        print("conv shape", conv2.shape)
        pool1 = TimeDistributed(MaxPooling1D(pool_size=2))(conv2)
        print("pool1 shape", pool1.shape)
        flat = TimeDistributed(Flatten())(pool1)
        print("flat config", flat.shape)
        # store layers
        in_layers.append(inputs)
        out_layers.append(flat)
    # merge heads
    merged = concatenate(out_layers)
    recurrent_1 = LSTM(100, activation='relu', return_sequences=True)(merged)
    recurrent_2 = LSTM(100, activation='relu', return_sequences=False)(recurrent_1)
    dropout_1 = Dropout(0.2)(recurrent_2)
    outputs= Dense(n_features)(dropout_1)
    
    model = Model(inputs=in_layers, outputs=outputs)
	# compile model
    model.compile(loss='mse', optimizer='adam')
    model.fit(input_data, y_train, epochs=n_epochs, batch_size=n_batch)
    print("end compile")
    return model


def concatenate_features(df_list_1, df_list_2):
    list_result = list()
    for i in range(0, len(df_list_1)):
        df_1 = df_list_1[i]
        df_2 = df_list_2[i]
        df_2 = df_2.drop(["date"], axis=1)
        df_2.columns = ["value_2"]
        result = pd.concat([df_1, df_2], axis=1, sort=False)
        list_result.append(result)
        
    return list_result
    

#config
#n_input: The number of lag observations to use as input to the model.
#n_filters: The number of parallel filters.
#n_kernel: The number of time steps considered in each read of the input sequence.
#n_epochs: The number of times to expose the model to the whole training dataset.
#n_batch: The number of samples within an epoch after which the weights are updated
def test():
    n_seq = 10
    n_steps = 96 
    n_filters = 1
    n_kernel = 20
    n_nodes = 100
    n_epochs = 20
    n_batch = 1
    n_input = n_seq * n_steps
    n_features = 1
    print("n_input", n_input)
    
    sequence, normal_sequence, anomalous_sequence = generate_sequences("12", "sensortgmeasurepp", limit=True, df_to_csv=True)
    sequence_2, normal_sequence_2, anomalous_sequence_2 = generate_sequences("11", "sensortgmeasurepp", limit=True, df_to_csv=True)
    list_result = concatenate_features(normal_sequence, normal_sequence_2)
    print("list_result", list_result)
    n_features = 2
    normal_sequence = list_result
    
    y_train_full = generate_full_y_train(normal_sequence, n_input, n_steps, n_features)
    print("y_train_full shape", y_train_full.shape)
    X_train_full = generate_full_X_train(normal_sequence, n_seq, n_input, n_steps, n_features)
    print("x_train_full shape", X_train_full.shape)
    #model = cnn_lstm(X_train_full, y_train_full, n_epochs, n_batch, n_kernel, n_filters, n_nodes, n_steps)

    model = multi_head(X_train_full, y_train_full, n_epochs, n_batch, n_kernel, n_filters, n_nodes, n_steps)
    history = list()
    number_of_chunks = 0
    for df_chunk in normal_sequence:
        number_of_chunks += 1
        print("number of chunks:", number_of_chunks)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
        X_train, y_train = generate_sets(df_chunk, n_seq, n_input, n_steps, n_features)
        print("X_train for train shape", X_train.shape)
        print("y_train for train shape", y_train.shape) 
        # fit
        history = model.fit(X_train, y_train, epochs=20, batch_size=n_batch, callbacks=[es, mc]).history
       

    return True