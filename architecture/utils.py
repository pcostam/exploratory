# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:35:26 2020

@author: anama
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.series import create_data, series_to_supervised, generate_sequences, generate_normal
import datetime
from sklearn.preprocessing import MinMaxScaler

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
def get_error_vector(X_val, X_pred):
    X_pred =  np.squeeze(X_pred)
    X_pred = X_pred[:,0]
    X_pred = X_pred.reshape(X_pred.shape[0],1)
        
    Xval =  np.squeeze(X_val)
    Xval = X_val[:,0]
    Xval = Xval.reshape(Xval.shape[0],1)
    x_input = Xval
    x_output = X_pred
    
    return np.abs(x_output - x_input)
    
 # calculate anormaly score (X-mu)^Tsigma^(-1)(X-mu)
def anomaly_score(mu, sigma, X):
    sigma_inv= np.linalg.inv(sigma)
    a = np.zeros(X.shape[0])
    for i in range(0, X.shape[0]):
            a[i] = (X[i] - mu).T*sigma_inv*(X[i] - mu)
    return a
    



def get_threshold(X_val_2_D, score):
    thresholds = [0.05, 0.5, 1, 2, 3]
    all_anormals = list()
    for th in thresholds:
        no_anomalous = 0
        i = 0
        for sc in score:
            if sc > th:
                no_anomalous += 1
                date = X_val_2_D['date'].iloc[i]
            i += 1
        print("no_anomalous", no_anomalous)
        all_anormals.append(no_anomalous)
    all_anormals = np.array(all_anormals)
    index_min = np.argmin(all_anormals)
    min_th = thresholds[index_min]
    return min_th

def plot_training_losses(history):
    fig, ax = plt.subplots(figsize=(14,6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mae)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.show()
    
    


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

def generate_full(raw, timesteps, model="LSTM", n_seq=None, n_input=None, n_features=None):  
    X_train_full = list()
    size = len(raw)
    if size  > 1:
        X_train_full = pd.concat(raw)
    else:
        print("normal_sequence", raw)
        print("size", len(raw))
        print(raw[0])
        X_train_full = raw[0]
        
    print("after concantening pieces", X_train_full)
    print(type(X_train_full))
    stime ="01-01-2017 00:00:00"
    etime ="01-03-2017 00:00:00"

    frmt = '%d-%m-%Y %H:%M:%S'
    min_date = datetime.datetime.strptime(stime, frmt)
    max_date = datetime.datetime.strptime(etime, frmt)
    print("min date", type(min_date))
    print("type", X_train_full.dtypes)

    X_train_full = preprocess(X_train_full, timesteps, put="in", model=model, n_seq=n_seq, n_input=n_input, n_features=n_features)
    print("X_train_full shape>>>", X_train_full.shape)
    if model == "LSTM":
        y_train_full = X_train_full[:, -1, :]
    else:
        y_train_full = generate_full_y_train(raw, n_input, timesteps, n_features)
    return X_train_full, y_train_full



def preprocess(raw, timesteps, put="in", model="LSTM", n_seq=None, n_input=None, n_features=None):
    if model == "CNN":
        if put == "in":
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
        elif put == "out":
            print("n input", n_input)
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
        
    if model == "LSTM":
        if put == "in" or put == "out":
            raw = raw.drop(['date'], axis = 1)
            data = np.array(raw['value'])
            data = series_to_supervised(raw, n_in=timesteps)
            data = np.array(data.iloc[:, :timesteps])
    
            #normalize data
            scaler = MinMaxScaler()
            data = scaler.fit_transform(data)
    
            #for lstm there is the need to reshape de 2-d np array to a 3-d np array [samples, timesteps, features]
            data = data.reshape((data.shape[0], data.shape[1],1))
    
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
  

    print("type", y_train_full.dtypes)
    
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


def generate_sets(raw, timesteps, type_input="LSTM", validation=True, n_seq=None, n_input=None, n_features=None):
    #CNN input 4D
    if type_input == "CNN": 
        y_train = preprocess(raw, timesteps, put="out", model="CNN", n_input=n_input, n_features=n_features)
        print("y_train shape", y_train.shape)
       
        X_train = preprocess(raw, timesteps, put="in", model="CNN", n_seq=n_seq, n_input=n_input, n_features=n_features)
        
        print("type xtrain", type(X_train))
        print("type ytrain", type(y_train))
        if validation == True:
            return generate_validation(raw, timesteps, model="CNN", n_seq=n_seq, n_input=n_input, n_features=n_features)
            
        return X_train, y_train, None, None, None, None
    
    #LSTM input 3D
    if type_input == "LSTM":
        normal_sequence = raw
        print("normal_sequence", normal_sequence)
        print("colunas", normal_sequence.shape[0])
    
        X_train_D = normal_sequence
    
        
        if validation == True:
            return generate_validation(X_train_D, timesteps)
            
        
        X_train = preprocess(X_train_D, timesteps)
        y_train = X_train[:,-1,:]
        #y_train = X_train[:, -1]
        
        return X_train, y_train, None, None

def split_features(n_features, X_train):
    input_data = list()
 
    for i in range(n_features):
        aux = X_train[:, :, :, i]
        print("aux shape", aux.shape)
        reshaped = aux.reshape((aux.shape[0], aux.shape[1], aux.shape[2], 1))
        input_data.append(reshaped)
     
        print("reshaped", reshaped.shape)
        
    return input_data

#LSTM
def generate_sets_days(normal_sequence, timesteps, validation=True):
    print("normal_sequence", type(normal_sequence))
    
    X_train_D = normal_sequence
    
    size_X_train_D = X_train_D.shape[0]
    size_train = round(size_X_train_D*0.8)
    if validation == True:
        X_train = X_train_D.iloc[:size_train, :]
        X_val = X_train_D.iloc[size_train:, :]
      
        size_val = round(0.5*X_val.shape[0])
       
        X_val_1_D = X_val.iloc[:size_val, :]
        X_val_2_D = X_val.iloc[size_val:, :]
    
        return X_train, X_val_1_D, X_val_2_D
    return X_train



def generate_validation(X_train_D, timesteps, model="LSTM",  n_seq=None, n_input=None, n_features=None):
     size_X_train_D = X_train_D.shape[0]
     size_train = round(size_X_train_D*0.8)
     
     X_train = X_train_D.iloc[:size_train, :]
     X_val = X_train_D.iloc[size_train:, :]
            
     size_val = round(0.5*X_val.shape[0])
       
     X_val_1_D = X_val.iloc[:size_val, :]
     X_val_2_D = X_val.iloc[size_val:, :]
     print("X_val_1 shape", X_val_1_D.shape)
     print("X_val_2 shape", X_val_2_D.shape)
    
     X_train = preprocess(X_train_D, timesteps, model=model, n_seq=n_seq, n_input=n_input, n_features=n_features)
     X_val_1 = preprocess(X_val_1_D, timesteps, model=model, n_seq=n_seq, n_input=n_input, n_features=n_features)
     X_val_2 = preprocess(X_val_2_D, timesteps, model=model, n_seq=n_seq, n_input=n_input, n_features=n_features)
            
           
     y_train = X_train[:,-1,:]
     y_val_1 = X_val_1[:,-1,:]
     y_val_2 = X_val_2[:, -1, :]
     print("X_train shape", X_train.shape)
     print("y_train shape", y_train.shape)
     
     return X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2
            
    