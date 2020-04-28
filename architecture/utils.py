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
import base64
from io import BytesIO
import seaborn as sns

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
    fo = open("demofile4.txt", "w")
    for el in X_pred:
        fo.write("X_pred " +  str(el))
    fo.close()
    X_val = process_predict(X_val)
    X_pred = process_predict(X_pred)   
    
    print("Xval shape", X_val.shape)
    print("X_pred shape", X_pred.shape)
    f = open("demofile3.txt", "w")
    for el_pred, el_val in zip(X_pred, X_val):
        f.write("X_pred " +  str(el_pred) + "\n")
        f.write("X_val" + str(el_val) + "\n")
    f.close()
    x_input = X_val
    x_output = X_pred
    
    return np.abs(x_output - x_input)
    
 # calculate anormaly score (X-mu)^Tsigma^(-1)(X-mu)
def anomaly_score(mu, sigma, X):
    sigma_inv= np.linalg.inv(sigma)
    a = np.zeros(X.shape[0])
    for i in range(0, X.shape[0]):
            a[i] = (X[i] - mu).T*sigma_inv*(X[i] - mu)
    return a
    

def process_input(X_train, y_train):
    if len(X_train.shape)==3:
        return X_train
          
    else:
        Xtrain = y_train
    return Xtrain


def get_threshold(X_val_2_D, score):
    thresholds = [x for x in range(0, 20)]
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
       X_pred = X_pred[:,-1]
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
    
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    
    #plt.show()
    
    return encoded
    
def plot_bins_loss(y_pred, ytrain, scored):
    scored['Loss_mae'] = np.mean(np.abs(y_pred-ytrain), axis = 1)
    figure = plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    sns.distplot(scored['Loss_mae'], bins=20, kde=True, color='blue')
    #plt.show()
    tmpfile = BytesIO()
    figure.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return encoded




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
    print("generate_full")
    X_train_full = preprocess(raw, timesteps, form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
    print("X_train_full shape>>>", X_train_full.shape)
    if output_form == "3D":
        y_train_full = X_train_full[:, -1, :]
    else:
        y_train_full = generate_full_y_train(raw, n_input, timesteps, n_features)
    return X_train_full, y_train_full



def preprocess(raw, timesteps, form="3D", input_data=pd.DataFrame(), n_seq=None, n_input=None, n_features=None):
    if form == "4D":
        raw = raw.drop(['date'], axis = 1)
        data = series_to_supervised(raw, n_in=n_input)
        print("shape data", data.shape)
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
        if len(input_data.shape) == 3:
            y_train = input_data[:, -1, :]
        else:
            raw = raw.drop(['date'], axis=1)
            data = series_to_supervised(raw, n_in=n_input)
            data = data.values
      
            y_train = data[:, :n_features]

            scaler = MinMaxScaler()
            y_train = scaler.fit_transform(y_train)

            y_train =  np.squeeze(y_train)
            y_train = np.reshape(y_train, (y_train.shape[0], n_features))
         
        return y_train
    
    elif form == "3D":
        raw = raw.drop(['date'], axis = 1)
        data = series_to_supervised(raw, n_in=timesteps)
        data = np.array(data.iloc[:, :timesteps])
    
        #normalize data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        #for lstm there is the need to reshape de 2-d np array to a 3-d np array [samples, timesteps, features]
        data = data.reshape((data.shape[0], data.shape[1],1))

        return data


def generate_full_y_train(normal_sequence, n_input, timesteps, n_features):
    y_train_full = normal_sequence.drop(['date'], axis=1)
    data = series_to_supervised(y_train_full, n_in=n_input)
   
    data = data.values
    
    if n_features == 1:
        y_train_full = [data[:, -1]]
    else:
        y_train_full = data[:, :n_features]
   
    
    scaler = MinMaxScaler()
    y_train_full = scaler.fit_transform(y_train_full)
    
    y_train_full =  np.squeeze(y_train_full)
    
    return y_train_full


def generate_sets(raw, timesteps,input_form ="3D", output_form = "3D", validation=True, n_seq=None, n_input=None, n_features=None):       
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
        reshaped = aux.reshape((aux.shape[0], aux.shape[1], aux.shape[2], 1))
        input_data.append(reshaped)
        
    return input_data

#LSTM
def generate_sets_days(normal_sequence, timesteps, validation=True):
    X_train_D = normal_sequence
    
    size_X_train_D = X_train_D.shape[0]
    size_train = round(size_X_train_D*0.8)
    if validation == True:
        X_train = X_train_D.iloc[:size_train, :]
        X_val = X_train_D.iloc[size_train:, :]
      
        size_val = round(0.5*X_val.shape[0])
       
        X_val_1_D = X_val.iloc[:size_val, :]
        print("X_val_1_D min days", min(X_val_1_D['date']))
        print("X_val_1_D max days", max(X_val_1_D['date']))
        X_val_2_D = X_val.iloc[size_val:, :]
        print("X_val_2_D days min", min(X_val_2_D['date']))
        print("X_val_2_D max days", max(X_val_2_D['date']))
        
    
        return X_train, X_val_1_D, X_val_2_D
    return X_train



def generate_validation(X_train_D, timesteps,input_form="3D", output_form="3D",  n_seq=None, n_input=None, n_features=None):
     size_X_train_D = X_train_D.shape[0]
     print("size_X_train_D", size_X_train_D)
     size_train = round(size_X_train_D*0.8)
     
     X_train = X_train_D.iloc[:size_train, :]
     X_val = X_train_D.iloc[size_train:, :]
            
     size_val = round(0.5*X_val.shape[0])
     print("size_val", size_val)
       
     X_val_1_D = X_val.iloc[:size_val, :]
     X_val_2_D = X_val.iloc[size_val:, :]
     print("shape x_val", X_val_1_D.shape)
    
     X_train = preprocess(X_train_D, timesteps,  form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     X_val_1 = preprocess(X_val_1_D, timesteps,  form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     X_val_2 = preprocess(X_val_2_D, timesteps,  form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
            
     
     y_train = preprocess(X_train_D, timesteps, input_data = X_train, form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     y_val_1 = preprocess(X_val_1_D, timesteps, input_data = X_val_1, form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     y_val_2 = preprocess(X_val_2_D, timesteps, input_data = X_val_2, form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     
     print("X_val_1", X_val_1.shape)
     print("y_val_1", y_val_1.shape)
     
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

def detect_anomalies(X_test, y_test, X_test_D, h5_filename, choose_th=None):
    param = load_parameters(h5_filename)

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
    
    
    #Xtest = preprocess(X_test, timesteps)
  
    print("Predict")
    y_pred = model.predict(X_test)

    predict = pd.DataFrame()
    
    vector = get_error_vector(y_test, y_pred)
    vector = np.squeeze(vector)
        
    score = anomaly_score(mu, sigma, vector)
    
    values = list()
    dates = list()
    anomalies = 0
    i = 0
    for sc in score:
        if sc > anomalies_th:
             anomalies += 1
             value = X_test_D['value'].iloc[i]
             date = X_test_D['date'].iloc[i]
             dates.append(date)
             values.append(value)
        i += 1
        
    print("no. anomalies", anomalies)
    print("predict", predict)
    
    predict['value'] = values
    predict['date'] = dates
    return predict

def test_2d():
    col_1 = [x for x in range(20)]
    col_2 = [x for x in range(20)]
    d = {'value': col_1, 'date':col_2}
    df = pd.DataFrame(data=d)
    y_test = preprocess(df, 2, form="2D", input_data=pd.DataFrame(), n_seq=2, n_input=4, n_features=1)
    print("y_test shape", y_test.shape)
    #comeu 4 porque e' o n_input e o n_input nao considera os nans
    return True

    