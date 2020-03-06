# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:42:20 2020

@author: anama
"""
from preprocessing.series import create_data
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import numpy as np
#see https://machinelearningmastery.com/how-to-develop-deep-learning-models-for-univariate-time-series-forecasting/
#https://towardsdatascience.com/get-started-with-using-cnn-lstm-for-forecasting-6f0f4dde5826
#https://blog.keras.io/building-autoencoders-in-keras.html
def cnn_lstm(X_train, y_train, n_epochs, n_batch, n_kernel, n_filters, n_nodes, n_steps):
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu', input_shape=(None,n_steps,1))))
    model.add(TimeDistributed(Conv1D(filters=n_filters, kernel_size=n_kernel, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    
    
    model.add(LSTM(n_nodes, activation='relu'))
    model.add(Dense(n_nodes, activation='relu'))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
	# fit
    model.fit(X_train, X_train, epochs=n_epochs, verbose=0)
    
    return model
def test():
  
    n_seq = 100 #3 months (meaning 4 sequences)
    n_steps = 3  #weeks? days? minutes? weeks.
    n_filters = 32
    n_kernel = 3
    n_nodes = 100
    n_epochs = 200
    n_batch = 20
    
    n_input = n_seq * n_steps
    print("n_input", n_input)
    data = create_data("sensortgmeasurepp", "12", n_input, limit=True)
    data = data.values
    print("data", type(data))
    print("data", data)
    
    X_train  = data[:, :-1]
    y_train = data[:, -1]
    print(X_train.shape)
 
    
    #[samples, subsequences, timesteps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], n_seq, n_steps, 1))
    
    cnn_lstm(X_train, y_train, n_epochs, n_batch, n_kernel, n_filters, n_nodes, n_steps)


    return True