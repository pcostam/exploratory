# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:30:29 2020

@author: anama
"""


from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from keras.models import Sequential
from keras.models import Model
from preprocessing.series import create_dataset_as_supervised, create_dataset, generate_sequences, series_to_supervised
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

#see https://medium.com/@shivajbd/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
def autoencoder_model(X):
    timesteps = X.shape[1]
    n_features = X.shape[2]
    model = Sequential()
    # Encoder
    model.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    # Decoder
    model.add(LSTM(16, activation='relu', return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True)) 
    model.add(TimeDistributed(Dense(n_features)))
 
    return model

def get_mu(vector):
    return np.mean(vector, axis=0)

def get_sigma(vector, mu):
     mu_T = np.array([mu], dtype=np.float32).T
     cov = np.zeros((mu.shape[0], mu.shape[0]))
     for e_i in vector:
            e_i_T = np.array([e_i], dtype=np.float32).T
            sig = np.dot((e_i_T-mu_T), (e_i_T-mu_T).T)
            cov += sig
     sigma = cov / vector.shape[0]
     print(sigma)
     return sigma
    
#https://scipy-lectures.org/intro/numpy/operations.html
def get_error_vector(x_input, x_output):
    return np.abs(x_output - x_input)
    
 # calculate anormaly score (X-mu)^Tsigma^(-1)(X-mu)
def anomaly_score(mu, sigma, X):
    sigma_inv= np.linalg.inv(sigma)
    a = np.zeros(X.shape[0])
    for i in range(0, X.shape[0]):
            a[i] = (X[i] - mu).T*sigma_inv*(X[i] - mu)
    return a
    
def plot_training_losses(history):
    fig, ax = plt.subplots(figsize=(14,6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mae)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.show()
    
#see https://towardsdatascience.com/lstm-autoencoder-for-anomaly-detection-e1f4f2ee7ccf
#https://towardsdatascience.com/lstm-autoencoder-for-extreme-rare-event-classification-in-keras-ce209a224cfb
def test_autoencoder():
    
    sequence, normal_sequence, anomalous_sequence = generate_sequences("12", "sensortgmeasurepp", limit=True)
    
    X_train_D, _, X_val, _ = train_test_split(normal_sequence, normal_sequence, test_size=0.4, random_state=1)
    
    X_train = X_train_D.drop(['date'], axis = 1)  
    
    #timesteps should be size of a week
    timesteps=3
  
    
    X_val_1_D, _, X_val_2_D, _ = train_test_split(X_val, X_val, test_size=0.5, random_state=1)
      
    X_val_1 = X_val_1_D.drop(['date'], axis = 1)
    X_val_2 = X_val_2_D.drop(['date'], axis = 1)
    
    data = series_to_supervised(X_val_1, n_in=timesteps)
    X_val_1 = np.array(data.iloc[:,:timesteps])
    
    data = series_to_supervised(X_val_2, n_in=timesteps)
    X_val_2 = np.array(data.iloc[:,:timesteps])
 
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
    
    """
    df = create_dataset("sensortgmeasurepp", "12")
    df = df.drop(['date','sensortgId', 'id'], axis = 1)
    Xtrain, Xtest, ytrain, ytest = create_dataset_as_supervised("sensortgmeasurepp", "12", limit=False)
    
    y_test = np.array(ytest["value"]).reshape(-1,1)
    y_train = np.array(ytrain["value"]).reshape(-1,1)
    X_train = Xtrain.loc[:,["var(t-3)", "var(t-2)", "var(t-1)"]]
    X_test = Xtest.loc[:, ["var(t-3)", "var(t-2)", "var(t-1)"]]
    """
    
    
    print("X_train", X_train)
    print("sequence", sequence)
    print("normal sequence", normal_sequence)
    print("anomalous sequence", anomalous_sequence)
    
    
    data = series_to_supervised(X_train, n_in=timesteps)
    X_train = np.array(data.iloc[:,:timesteps])
    print("X_train", X_train)
    #normalize data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    
    
    #for lstm there is the need to reshape de 2-d np array to a 3-d np array [samples, timesteps, features]
    #X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
    X_val_1 = X_val_1.reshape((X_val_1.shape[0], X_val_1.shape[1],1))
    X_val_2 = X_val_2.reshape((X_val_2.shape[0], X_val_2.shape[1],1))
    #print("X_test shape", X_test.shape)
    print("X_train shape", X_train.shape)
    print(X_train)
    model = autoencoder_model(X_train)
    model.compile(optimizer='adam', loss='mae')
    model.summary()
   
    nb_epochs = 25
    batch_size = 250
    
    history = model.fit(X_train, X_train, validation_data=(X_val_1, X_val_1), epochs=nb_epochs, batch_size=batch_size, validation_split=0.05, callbacks=[es, mc]).history
    
    # load the saved model
    saved_model = load_model('best_model.h5')
    
    
    plot_training_losses(history)
    
    X_pred = saved_model.predict(X_train)
    print("shape pred:", X_pred.shape)
    print(X_pred)
    
    
    X_pred = np.squeeze(X_pred)
    X_pred = X_pred[:,0]
    X_pred = X_pred.reshape(X_pred.shape[0], 1)
    print("shape pred:", X_pred.shape)
    
    X_pred = pd.DataFrame(X_pred)
    
    scored = pd.DataFrame(index=X_pred.index)

    Xtrain =  np.squeeze(X_train)
    Xtrain = Xtrain[:,0]
    Xtrain = Xtrain.reshape(Xtrain.shape[0],1)
    print("shape train:", X_train.shape)
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
    plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    sns.distplot(scored['Loss_mae'], bins=20, kde=True, color='blue')
    plt.show()
    
    #calculate loss on the validation set to get miu and sigma values
    
    X_pred = model.predict(X_val_1)
    
    X_pred =  np.squeeze(X_pred)
    X_pred = X_pred[:,0]
    X_pred = X_pred.reshape(X_pred.shape[0],1)
    
    Xval1 =  np.squeeze(X_val_1)
    Xval1 = X_val_1[:,0]
    Xval1 = Xval1.reshape(Xval1.shape[0],1)
    
    
    vector = get_error_vector(Xval1, X_pred)
    vector = np.squeeze(vector)
    
    plt.hist(list(vector), bins=20)
    plt.show()
    
    vector = vector.reshape(vector.shape[0], 1)
    print("vector shape", vector.shape)
    print(vector)
    
    mu = get_mu(vector)
    print("mu", mu)
    sigma = get_sigma(vector, mu)
    print("sigma", sigma)
    
    score = anomaly_score(mu, sigma, vector)
    print(score)
    
    X_pred = model.predict(X_val_2)
    
    X_pred =  np.squeeze(X_pred)
    X_pred = X_pred[:,0]
    X_pred = X_pred.reshape(X_pred.shape[0],1)
    
    
    Xval2 =  np.squeeze(X_val_2)
    Xval2 = X_val_2[:,0]
    Xval2 = Xval2.reshape(Xval2.shape[0],1)
    
    vector = get_error_vector(Xval2, X_pred)
    vector = np.squeeze(vector)
    
    score = anomaly_score(mu, sigma, vector)
    thresholds = [0.05, 0.5, 1,2]
    
    for th in thresholds:
        no_anomalous = 0
        i = 0
        for sc in score:
            if sc > th:
                no_anomalous += 1
                date = X_val_2_D['date'].iloc[i]
                print("date", date)
            i += 1
        print("no_anomalous", no_anomalous)
    
    
    
    
    
    
    
    
    return True