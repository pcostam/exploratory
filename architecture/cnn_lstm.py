# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:42:20 2020

@author: anama
"""
from preprocessing.series import create_data, series_to_supervised, generate_sequences, generate_normal
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
#see https://machinelearningmastery.com/how-to-develop-deep-learning-models-for-univariate-time-series-forecasting/
#https://towardsdatascience.com/get-started-with-using-cnn-lstm-for-forecasting-6f0f4dde5826
#https://blog.keras.io/building-autoencoders-in-keras.html
def cnn_lstm(X_train, y_train, n_epochs, n_batch, n_kernel, n_filters, n_nodes, n_steps):
    n_features = X_train.shape[3]
    print("n_features", 1)
    size = n_steps*n_filters
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu'), input_shape=(None,n_steps,1)))
    model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten())) 
    #model.add((Dense(3)))
    #model.add(Reshape((1,2)))
    model.add(LSTM(n_nodes, activation='relu', return_sequences=True))
    model.add(LSTM(n_nodes, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(n_features))
 
    
    #model.add(TimeDistributed(Dense(n_features)))
    #model.add(LSTM(50, activation='relu', return_sequences=True))
    #TimeDistributed because what is wanted is to predict a sequence
    #model.add(TimeDistributed(Dense(n_features)))
   
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    model.compile(loss='mse', optimizer='adam')
  
    print(model.summary())
    outputs = [layer.output for layer in model.layers]
    inputs = [layer.input for layer in model.layers]
    print("outputs", outputs)
    print("inputs", inputs)
    
  
 
	# fit
    model.fit(X_train, y_train, batch_size=n_batch, epochs=n_epochs, verbose=0)
    
    return model
def test():
  
    n_seq = 300 #3 months (meaning 4 sequences)
    n_steps = 3 #weeks? days? minutes? weeks.
    n_filters = 2
    n_kernel = 1
    n_nodes = 100
    n_epochs = 200
    n_batch = 1
    
    n_input = n_seq * n_steps
    #n_input = 100
    print("n_input", n_input)
    
    sequence, normal_sequence, anomalous_sequence = generate_sequences("12", "sensortgmeasurepp", limit=True, df_to_csv=True)
    normal_sequence = normal_sequence.drop(['date'], axis=1)
    data = series_to_supervised(normal_sequence, n_in=n_input)
    
    print("data", data)
   
    data = data.values
    print("data", type(data))
    print("data", data)
    
    X_train  = data[:, :-1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
 
 
  
    y_train = [data[:, -1]]
 
    print("y_train", y_train)
  
    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_train)
    
   

    print("y_train shape", y_train.shape)
    y_train =  np.squeeze(y_train)
    #y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
    #[batch_size, height, width, depth]
    #[samples, subsequences, timesteps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], n_seq, n_steps, 1))
    print("X_train shape", X_train.shape)
    
    

    
   
    
    cnn_lstm(X_train, y_train, n_epochs, n_batch, n_kernel, n_filters, n_nodes, n_steps)


    return True