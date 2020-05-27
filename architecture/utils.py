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
def get_error_vector(x_input, x_output, timesteps, n_features): 
    x_input = process_predict(x_input, timesteps, n_features)
    x_output = process_predict(x_output, timesteps, n_features)   
    
    vector = np.abs(x_output - x_input)
    vector = np.squeeze(vector)
    if len(vector.shape) == 1:
        vector = vector.reshape(vector.shape[0], 1)
    return vector
    
 # calculate anormaly score (X-mu)^Tsigma^(-1)(X-mu)
def anomaly_score(mu, sigma, X, n_features):
    # D- number of features
    #sigma DxD matrix
    #mu vector D-dimensional
    #X- error vector D-dimensional
    X = X.reshape(X.shape[0], n_features)
    sigma_inv= np.linalg.inv(sigma)
    a = np.zeros(X.shape[0])
    for i in range(0, X.shape[0]):
            x = X[i, :]
            a[i] = (x - mu).T@sigma_inv@(x - mu)              
    return a
    

    

def process_input(X_train, y_train):
    if len(X_train.shape)==3:
        return X_train
          
    else:
        Xtrain = y_train
    return Xtrain


def get_threshold(dates, score):
    thresholds = [x for x in range(0, 20)]
    all_anormals = list()
    for th in thresholds:
        no_anomalous = 0
        i = 0
        for sc in score:
            if sc > th:
                no_anomalous += 1
            i += 1
        print("no_anomalous", no_anomalous)
        all_anormals.append(no_anomalous)
    all_anormals = np.array(all_anormals)
    index_min = np.argmin(all_anormals)
    min_th = thresholds[index_min]
    return min_th

def process_predict(X_pred,  timesteps, n_features): 
    print("n_features", n_features)
    print("timesteps", timesteps)
    print("X_pred shape 1", X_pred.shape)
    if len(X_pred.shape) == 3:
        X_pred = X_pred[:,:n_features,:]
        X_pred = X_pred.reshape(X_pred.shape[0], n_features)
        print("X_pred shape", X_pred.shape)
    else:
        X_pred = X_pred.reshape(X_pred.shape[0], n_features)
    print("X_pred.shape 4", X_pred.shape)
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
    
def plot_bins_loss(loss):
    figure = plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    sns.distplot(loss, bins=20, kde=True, color='blue')
    #plt.show()
    tmpfile = BytesIO()
    figure.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return encoded

def plot_series(title, dates, y_true, y_pred, timesteps, n_features):
    y_true = process_predict(y_true, timesteps, 1)
    y_pred = process_predict(y_pred, timesteps, 1)   
    fig, ax = plt.subplots(figsize=(14,6), dpi=80)
    plt.plot(dates, y_true, 'b', label="Validation")
    plt.plot(dates, y_pred, 'r', label="Prediction")
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Value')
    ax.set_xlabel('Dates')
    ax.legend(loc='upper right')
    #plt.show()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    return encoded


  
def concatenate_features(columns):
    agg = pd.concat(columns, axis=1, sort=False)
    agg =  agg.drop_duplicates("date", "first")
    col = (agg.columns == 'date').argmax()
    dates = agg.iloc[:, col]
    agg = agg.drop(['date'], axis=1)
    agg['date'] = dates                 
    return agg

def generate_full(raw, timesteps,input_form="3D", output_form="3D", n_seq=None, n_input=None, n_features=None):        
    X_train_full = preprocess(raw, timesteps, form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
 
    if output_form == "3D":
        y_train_full = X_train_full[:, :, :n_features]
    else:
        y_train_full = generate_full_y_train(raw, n_input, timesteps, n_features)
    return X_train_full, y_train_full



def preprocess(raw, timesteps, form="3D", input_data=pd.DataFrame(), n_seq=None, n_input=None, n_features=None, dates=False):   
    if isinstance(raw, pd.DataFrame) and dates == False:
        data = series_to_supervised(raw, n_in=n_input)
    else:
        data = raw
        
    
    data = preprocess_shapes(data, timesteps, form, input_data=input_data, n_seq=n_seq, n_input=n_input, n_features=n_features, dates=dates)
    
    return data
def preprocess_shapes(data, timesteps, form="3D", input_data=pd.DataFrame(), n_seq=None, n_input=None, n_features=None, dates=False):
    if form == "4D":
        print("n_input", n_input)
        data = np.array(data.iloc[:, :n_input*n_features])
        print("data shape", data.shape)
        #normalize data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        #for CNN there is the need to reshape de 2-d np array to a 4-d np array [samples, timesteps, features]
        #[batch_size, height, width, depth]
        #[samples, subsequences, timesteps, features]
        columns = data.shape[1]
        samples = data.shape[0]
        cells = samples * columns
        new_n_seq = round(cells/(samples*timesteps*n_features))
        print("n_seq", n_seq)
        if new_n_seq != n_seq:
            print('Not possible to generate this number of sequences: %s. New sequence %s' %(n_seq, new_n_seq))
            n_seq = new_n_seq
        print("samples", samples)
        print("n_seq", n_seq)
        print("timesteps", timesteps)
        print("n_features", n_features)
        data = np.reshape(data, (samples, n_seq, timesteps, n_features))
    
        return data
       
    elif form == "2D":
        if len(input_data.shape) == 3:
            print("2D input_data", input_data.shape)
            y_train = input_data[:, :n_features, :]
        else:
            data = data.values
      
            y_train = data[:, :n_features]

            scaler = MinMaxScaler()
            y_train = scaler.fit_transform(y_train)

            y_train =  np.squeeze(y_train)
            y_train = np.reshape(y_train, (y_train.shape[0], n_features))
         
        return y_train
    
    elif form == "3D":
        data = np.array(data.iloc[:, :timesteps*n_features])
        print("data", data.shape)
    
        #normalize data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        #for lstm there is the need to reshape de 2-d np array to a 3-d np array [samples, timesteps, features]
        data = data.reshape((data.shape[0], timesteps, n_features))

        return data


def generate_full_y_train(normal_sequence, n_input, timesteps, n_features):
    data = series_to_supervised(normal_sequence, n_in=n_input)
   
    data = data.values
    
    if n_features == 1:
        y_train_full = [data[:, -1]]
    else:
        y_train_full = data[:, :n_features]
   
    
    scaler = MinMaxScaler()
    y_train_full = scaler.fit_transform(y_train_full)
    
    y_train_full =  np.squeeze(y_train_full)
    
    return y_train_full


def generate_sets(raw, timesteps,input_form ="3D", output_form = "3D", validation=True, n_seq=None, n_input=None, n_features=None, dates=False):       
    print(">>>>generate_sets")
    print("N_features", n_features)
    if validation:
        return generate_validation(raw, timesteps, input_form=input_form, output_form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features, dates=dates)

    X_train = preprocess(raw, timesteps, form = input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
  
    if output_form == "3D":
        y_train = X_train
    if output_form == "2D":
        y_train = preprocess(raw, timesteps, form = output_form, input_data = X_train, n_seq=n_seq, n_input=n_input, n_features=n_features)
    print("generate_sets y_train shape", y_train.shape)
    print("generte_sets X_train shape", X_train.shape)
    return X_train, y_train, None, None, None, None
    

def split_features(n_features, X_train):
    print("split_features")
    print("n_features", n_features)
    input_data = list()
 
    for i in range(n_features):
        aux = X_train[:, :, :, i]
        reshaped = aux.reshape((aux.shape[0], aux.shape[1], aux.shape[2], 1))
        input_data.append(reshaped)
        
    return input_data



 
def generate_validation(X_train_D, timesteps,input_form="3D", output_form="3D",  n_seq=None, n_input=None, n_features=None, dates=False):
     print("X_train_D")
     X_train = series_to_supervised(X_train_D, n_in=n_input, dates=dates)
             
     size_X_train = X_train.shape[0]
     size_train = round(size_X_train*0.8)
     X_val = X_train.iloc[size_train:, :]
     X_train = X_train.iloc[:size_train, :]
     size_val = round(0.5*X_val.shape[0])
       
     X_val_1 = X_val.iloc[:size_val, :]
     X_val_2 = X_val.iloc[size_val:, :]
     
        
     Xtrain = preprocess_shapes(X_train, timesteps,  form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     Xval_1 = preprocess_shapes(X_val_1, timesteps,  form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     Xval_2 = preprocess_shapes(X_val_2, timesteps,  form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
            
     
     y_train = preprocess_shapes(X_train, timesteps, input_data = X_train, form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     y_val_1 = preprocess_shapes(X_val_1, timesteps, input_data = X_val_1, form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     y_val_2 = preprocess_shapes(X_val_2, timesteps, input_data = X_val_2, form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     
     print("X_val_1", X_val_1.shape)
     print("y_val_1", y_val_1.shape)
     
     return Xtrain, y_train, Xval_1, y_val_1, Xval_2, y_val_2
 
def save_parameters(mu, sigma, timesteps, min_th, filename):
    print("save parameters")
    param = {'mu':mu, 'sigma':sigma, 'timesteps':timesteps, 'min_th':min_th}
    filename = 'parameters_' + filename + '.pickle'
    path = os.path.join("..//gui_margarida//gui/assets", filename)
    with open(path, 'wb') as f:  
        pickle.dump(param, f, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def load_parameters(filename):
    global mu, sigma, timesteps, min_th
    filename = 'parameters_' + filename + '.pickle'
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

def get_date(df):
    return df['date']


def generate_days(X_train_D, n_input, time, validation=True):   
    X_train = pd.DataFrame()
    X_train = series_to_supervised(X_train_D, n_in=n_input, dates=True)
    if validation == True:   
       
        size_X_train = X_train.shape[0]
       
        size_train = round(size_X_train*0.8)
        X_val = X_train.iloc[size_train:, :]
        X_train = X_train.iloc[:size_train, :]
       
        size_val = round(0.5*X_val.shape[0])
        print("size_val", size_val)
          
        X_val_1 = X_val.iloc[:size_val, :]
        X_val_2 = X_val.iloc[size_val:, :]
        
       
    
        return X_train[time], X_val_1[time], X_val_2[time]
    else:
        return X_train[time]

def drop_date(df):
    df = df.drop(['date'], axis = 1)
    return df

def detect_anomalies(X_test, y_test, X_test_D, h5_filename, timesteps, n_features, choose_th=None, time='date'):
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
      
    print("Predict")
    y_pred = model.predict(X_test)

    predict = pd.DataFrame()
    
    print("y_test shape", y_test.shape)
    print("y_pred shape", y_pred.shape)
    
    vector = get_error_vector(y_test, y_pred, timesteps, n_features)
    vector = np.squeeze(vector)
        
    score = anomaly_score(mu, sigma, vector, n_features)
    
    values = list()
    dates = list()
    anomalies = 0
    i = 0
    for sc in score:
        if sc > anomalies_th:
             anomalies += 1
             value = X_test_D['value'].iloc[i]
             date = X_test_D[time].iloc[i]
             dates.append(date)
             values.append(value)
        i += 1
        
    print("no. anomalies", anomalies)
    print("predict", predict)
    
    predict['value'] = values
    predict[time] = dates
    print("end")
    return predict

def join_partitions_features(train_chunks_all, no_partitions_cv,to_exclude):
    """Concatenates information of several sensors"""
    train_chunks = list()
    for i in range(0,no_partitions_cv):
        to_concat = [item[i] for item in train_chunks_all] 
        union_normal_sequences = pd.concat(to_concat, axis=1) 
        union_normal_sequences =  union_normal_sequences.drop_duplicates(to_exclude, "first")
        for column in to_exclude:
            col = (union_normal_sequences.columns == column).argmax()
            dates = union_normal_sequences.iloc[:, col]
            union_normal_sequences = union_normal_sequences.drop([column], axis=1)
            union_normal_sequences[column] = dates                 
        train_chunks.append(union_normal_sequences)
    return train_chunks
          

def test_2d():
    col_1 = [x for x in range(20)]
    col_2 = [x for x in range(20)]
    d = {'value': col_1, 'date':col_2}
    df = pd.DataFrame(data=d)
    y_test = preprocess(df, 2, form="2D", input_data=pd.DataFrame(), n_seq=2, n_input=4, n_features=1)
    print("y_test shape", y_test.shape)
    #comeu 4 porque e' o n_input e o n_input nao considera os nans
    return True

def test():
   col_1 = [x for x in range(20)]
   col_2 = [x for x in range(20)]
   col_3 = [x for x in range(20,40)]
   col_4 = [x for x in range(20)]
  
   d = {'value': col_1, 'date':col_2}
   d2 = {'value': col_3, 'date': col_4}
   df = pd.DataFrame(data=d)
   df2 = pd.DataFrame(data=d2)
   
   df_union = pd.concat([df, df2], axis=1)
   col = (df_union.columns == 'date').argmax()
   dates = df_union.iloc[:, col]
   df_other = df_union.drop(['date'], axis=1)
   df_other['date'] = dates
   print("dates", dates)
   #df_other['date'] = dates
   print("df.columns", df_union.columns)
   print("type", type(df_union.columns))
   print(df.columns.where(df.columns == 'date'))
  
   return df_other
   
