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
import os
import pickle
from keras.models import load_model

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
    Xval = process_predict(X_val)
    X_pred = process_predict(X_pred)     
  
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

def process_predict(X_pred):
    if len(X_pred.shape) == 3:
        X_pred = np.squeeze(X_pred)
        X_pred = X_pred[:,0]
        X_pred = X_pred.reshape(X_pred.shape[0], 1)
     
    return X_pred
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

def generate_full(raw, timesteps,input_form="3D", output_form="3D", n_seq=None, n_input=None, n_features=None):  
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

    X_train_full = preprocess(X_train_full, timesteps, form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
    print("X_train_full shape>>>", X_train_full.shape)
    if output_form == "3D":
        y_train_full = X_train_full[:, -1, :]
    else:
        y_train_full = generate_full_y_train(raw, n_input, timesteps, n_features)
    return X_train_full, y_train_full



def preprocess(raw, timesteps, form="3D", input_data=pd.DataFrame(), n_seq=None, n_input=None, n_features=None):
    print("preprocess")
    print("form 1", form)
    print("input_data shape", input_data.shape)
    if form == "4D":
        raw = raw.drop(['date'], axis = 1)
        data = series_to_supervised(raw, n_in=n_input)
        data = np.array(data.iloc[:, :n_input])
     
        #normalize data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        #for CNN there is the need to reshape de 2-d np array to a 4-d np array [samples, timesteps, features]
        #[batch_size, height, width, depth]
        #[samples, subsequences, timesteps, features]
        rows = data.shape[0] * data.shape[1]
        new_n_seq = round(rows/(data.shape[0]*timesteps*n_features))

        n_seq = new_n_seq
        data = np.reshape(data, (data.shape[0], n_seq, timesteps, n_features))
    
        return data
       
    elif form == "2D":
        print("input_data shape", input_data.shape)
        print("len input_data", len(input_data.shape))
        if len(input_data.shape) == 3:
            y_train = input_data[:, -1, :]
        else:
            print("2D OTHER")
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
            y_train = np.reshape(y_train, (y_train.shape[0], n_features))
            print("Y_TRAIN 2D SHAPE", y_train.shape)
        return y_train
    
    elif form == "3D":
        raw = raw.drop(['date'], axis = 1)
        #data = np.array(raw['value'])
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


def generate_sets(raw, timesteps,input_form ="3D", output_form = "3D", validation=True, n_seq=None, n_input=None, n_features=None):       
    print("generate_sets")
    print("n_input", n_input)
    print("output_form", output_form)
    print("input_form", input_form)
    print("validation generate_sets", validation)
    if validation == True:
        return generate_validation(raw, timesteps, input_form=input_form, output_form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
               
    X_train = preprocess(raw, timesteps, form = input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
  
    if output_form == "3D":
        y_train = X_train
    if output_form == "2D":
        y_train = preprocess(raw, timesteps, form = output_form, input_data = X_train, n_seq=n_seq, n_input=n_input, n_features=n_features)
    print("y_train shape", y_train.shape)
    print("X_train shape", X_train.shape)
    return X_train, y_train, None, None, None, None
    

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



def generate_validation(X_train_D, timesteps,input_form="3D", output_form="3D",  n_seq=None, n_input=None, n_features=None):
     print("generate validation")
     print("input form", input_form)
     print("output form", output_form)
     size_X_train_D = X_train_D.shape[0]
     size_train = round(size_X_train_D*0.8)
     
     X_train = X_train_D.iloc[:size_train, :]
     X_val = X_train_D.iloc[size_train:, :]
            
     size_val = round(0.5*X_val.shape[0])
       
     X_val_1_D = X_val.iloc[:size_val, :]
     X_val_2_D = X_val.iloc[size_val:, :]
     print("X_val_1 shape", X_val_1_D.shape)
     print("X_val_2 shape", X_val_2_D.shape)
     print("X_train_D", X_train_D.shape)
    
     X_train = preprocess(X_train_D, timesteps,  form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     X_val_1 = preprocess(X_val_1_D, timesteps,  form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     X_val_2 = preprocess(X_val_2_D, timesteps,  form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
            
     
     y_train = preprocess(X_train_D, timesteps, input_data = X_train, form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     y_val_1 = preprocess(X_val_1_D, timesteps, input_data = X_val_1, form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     y_val_2 = preprocess(X_val_2_D, timesteps, input_data = X_val_2, form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     

        
     print("X_train shape", X_train.shape)
     print("y_train shape", y_train.shape)
     
     return X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2
 
def save_parameters(mu, sigma, timesteps, min_th, filename):
    param = {'mu':mu, 'sigma':sigma, 'timesteps':timesteps, 'min_th':min_th}
    filename = 'parameters' + filename + '.pickle'
    path = os.path.join("..//gui_margarida//gui/assets", filename)
    with open(path, 'wb') as f:  
        pickle.dump(param, f, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def load_parameters(filename):
    global mu, sigma, timesteps, min_th
    filename = 'parameters' + filename + '.pickle'
    current_dir = os.getcwd()
    print("current dir", current_dir)
    path = ""
    if current_dir == "F:\\manual\\Tese\exploratory\\wisdom\\architecture":
        path = os.path.join("..//gui_margarida//gui//assets", filename)
    else:
        path = os.path.join(current_dir + '//assets//' + filename)
    
    # Getting back the objects:
    with open(path, 'rb') as f:  
        param = pickle.load(f)
    
    mu = param['mu']
    sigma = param['sigma']
    timesteps = param['timesteps']
    min_th = param['min_th']
    
    return param

def detect_anomalies(X_test, h5_filename, choose_th=None):
    print("shape", X_test.shape)
    param = load_parameters()

    filename = h5_filename + '.h5'
    current_dir = os.getcwd()
    print("current dir", current_dir)
    path = ""
    if current_dir == "F:\\manual\\Tese\exploratory\\wisdom\\architecture":
        path = os.path.join("..//gui_margarida//gui//assets", filename)
    else:
        path = os.path.join(current_dir + '//assets//' + filename)
        
    model = load_model(path) 
    
    mu = param['mu']
    sigma = param['sigma']
    timesteps = param['timesteps']
    print("mu", mu)
    print("sigma", sigma)
    print("timesteps", timesteps)
    
    anomalies_th = param['min_th']
    if choose_th != None:
        anomalies_th = choose_th
        
    print("treshold", anomalies_th)
    
    Xtest = preprocess(X_test, timesteps)
    print("Xtest shape", Xtest.shape)
    print("Xtest type", type(Xtest))
    
    print("Predict")
    X_pred = model.predict(Xtest)

    predict = pd.DataFrame()
    
    vector = get_error_vector(Xtest, X_pred)
    vector = np.squeeze(vector)
        
    score = anomaly_score(mu, sigma, vector)
    
    values = list()
    dates = list()
    anomalies = 0
    i = 0
    for sc in score:
        if sc > anomalies_th:
             anomalies += 1
             value = X_test['value'].iloc[i]
             date = X_test['date'].iloc[i]
             dates.append(date)
             values.append(value)
        i += 1
        
    print("no. anomalies", anomalies)
    print("predict", predict)
    
    predict['value'] = values
    predict['date'] = dates
    return predict

            
    