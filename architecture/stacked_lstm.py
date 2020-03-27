# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:15:38 2020

@author: anama
"""
#See references
#https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

def model(num_lstm_layers):
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    
    for i in range(num_lstm_layers):
        name = 'layer_lstm_encoder_{0}'.format(i+1)
        model.add(LSTM(50, activation='relu', return_sequences=True, name=name))
    
    model.add(Dense(1))
    
    adam = Adam(lr=learning_rate, decay= adam_decay)
    model.compile(optimizer=adam, loss='mae',  metrics=['accuracy'])
    model.summary()
    
    return model