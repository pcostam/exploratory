# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:31:53 2019

@author: anama
"""
from preprocessing.series import create_dataset_as_supervised
from architecture.lstm import model_with_quantiles, q_loss, model_method
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("test")
    X_train, X_test, y_train, y_test = create_dataset_as_supervised("sensortmmeasure", "12")
   
    print("len train", len(y_train))
    #for lstm there is the need to reshape de 2-d np array to a 3-d np array 
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],1))
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
    print("X_TEST SHAPE", X_test.shape)
    print("X_Train SHAPE", X_train.shape)
    
    model = model_with_quantiles(X_train.shape[1], X_train.shape[2])
    #https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/
    # The lambda function is used to input the quantile value to the quantile
    # regression loss function. Keras only allows two inputs in user-defined loss
    # functions, predictions and actual values.
    
    
    quantiles=[0.1, 0.5, 0.9]
    order = np.arange(len(y_test))
    plt.scatter(order, y_test)
 
    for q in quantiles:
        model.compile(optimizer='adam', loss = lambda y_pred,y_true : q_loss(q, y_pred, y_true))
       
        model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=2, shuffle=True)
       
        y_pred = model.predict(X_test)
       
        plt.plot(order, y_pred, label=q)
    plt.figure()
    plt.show()
    
    
    """
    print("method 2>>>>>")
    #METHOD 2
    print("X_TRAIN", X_train.shape)

    losses = [lambda y,f: q_loss(0.1,y,f), lambda y,f: q_loss(0.5,y,f), lambda y,f: q_loss(0.9,y,f)]
    model = model_method(X_train.shape[1], X_train.shape[2])
    model.compile(optimizer='adam', loss = losses, loss_weights=[0.3,0.3,0.3])
    model.fit(X_train, [y_train,y_train, y_train], epochs=100, batch_size=1, verbose=2, shuffle=True)
    predictions = model.predict(X_test)
    
    for i, prediction in enumerate(predictions):
        plt.plot(order, prediction, label='{}th Quantile'.format(int(quantiles[i]*100)))
    """
    
  
 

    
if __name__=="__main__":
    main()