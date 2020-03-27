# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:20:53 2020

@author: anama
"""


def model(X_train, y_train, n_epochs, n_batch, n_kernel, n_filters, n_nodes, n_steps, num_lstm_layers):
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
    model.add(Bidirectional(LSTM(n_nodes, activation='relu', return_sequences=True)))
    for i in range(num_lstm_layers):
        name = 'layer_lstm_decoder_{0}'.format(i+1)
        model.add(Bidirectional(LSTM(n_nodes, activation='relu', return_sequences=False, name=name)))
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