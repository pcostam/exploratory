# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:30:29 2020

@author: anama
"""


from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from keras.models import Sequential
from keras.models import Model
from preprocessing.series import create_dataset_as_supervised, create_dataset, generate_sequences, series_to_supervised, select_data, generate_normal
from preprocessing.series import downsample
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import skopt
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt.callbacks import DeltaYStopper
import tensorflow
from keras.backend import clear_session
from keras import regularizers
import datetime
from keras.optimizers import Adam
from architecture.evaluate import f_beta_score
import time

X_train = pd.DataFrame()
X_val_1 = pd.DataFrame()
config = [2, 128, 1e-2, 0.5, 0.5]

#see https://medium.com/@crawftv/parameter-hyperparameter-tuning-with-bayesian-optimization-7acf42d348e1
def hyperparam_opt():
   
    dim_num_lstm_layers = Integer(low=0, high=20, name='num_lstm_layers')
    dim_batch_size = Integer(low=64, high=128, name='batch_size')
    #dim_adam_decay = Real(low=1e-6,high=1e-2,name="adam_decay")
    dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
    dim_drop_rate_1 = Real(low=0.2 ,high=0.9,name="drop_rate_1")
    dim_drop_rate_2 = Real(low=0.2 ,high=0.9,name="drop_rate_2")
    dimensions = [dim_num_lstm_layers,
                  dim_batch_size,
                  dim_learning_rate,
                  dim_drop_rate_1,
                  dim_drop_rate_2]
    
    default_parameters = [2, 128, 1e-2, 0.5, 0.5]

    return dimensions,  default_parameters

dimensions,  default_parameters = hyperparam_opt()

#see https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html#skopt.gp_minimize
#https://www.kdnuggets.com/2019/06/automate-hyperparameter-optimization.html
@use_named_args(dimensions=dimensions)
def fitness(num_lstm_layers, batch_size, learning_rate, drop_rate_1, drop_rate_2):  
    init = time.perf_counter()
    print("fitness>>>")
    print("number lstm layers:", num_lstm_layers)
    print("batch size:", batch_size)
    print("learning rate:", learning_rate)
    print("drop rate 1:", drop_rate_1)
    print("drop rate 2:", drop_rate_2)
    
    _, normal_sequence, _ = generate_sequences("12", "sensortgmeasurepp", limit=True, df_to_csv=True)
    
   
    normal_sequence = generate_normal("12", limit=True, n_limit=129600, df_to_csv = True)
    X_train_full = generate_full_X_train(normal_sequence, 96)
    model = autoencoder_model(X_train_full, num_lstm_layers, learning_rate, drop_rate_1, drop_rate_2)
    
    print("total number of chunks", len(normal_sequence))
    no_chunks = 0
    for df_chunk in normal_sequence:
        no_chunks += 1
        print("number of chunks:", no_chunks)
        X_train, X_val_1, X_val_2 = generate_sets(df_chunk, 96) 
        es = EarlyStopping(monitor='val_loss', min_delta = 0.01, mode='min', verbose=1)
        hist = model.fit(X_train, X_train, validation_data=(X_val_1, X_val_1), epochs=100, batch_size= batch_size, callbacks=[es])
    
    
    #print(hist.history.keys())
    #accuracy = 0
    """
    try:
        accuracy = hist.history['val_accuracy'][-1]
    except KeyError:
        accuracy = hist.history['val_acc'][-1]
    """
    loss = hist.history['loss'][-1]
    
    #print("Accuracy: {0:.2%}".format(accuracy))
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    clear_session()
    tensorflow.compat.v1.reset_default_graph()
    
    end = time.perf_counter()
    diff = end - init
    # the optimizer aims for the lowest score, so we return our negative accuracy
    #return -accuracy
    return loss, diff
  
#see https://medium.com/@shivajbd/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
def autoencoder_model(X, num_lstm_layers, learning_rate, drop_rate_1, drop_rate_2):
    print("X", X)
    timesteps = X.shape[1]
    n_features = X.shape[2]
    model = Sequential()
    # Encoder  
    model.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True, kernel_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))
    for i in range(num_lstm_layers):
        name = 'layer_lstm_encoder_{0}'.format(i+1)
        model.add(LSTM(32, activation='relu', return_sequences=True, name=name))      
    model.add(LSTM(16, activation='relu',return_sequences=False))
    model.add(Dropout(rate=drop_rate_1))
    model.add(RepeatVector(timesteps))
    # Decoder
    model.add(LSTM(16, activation='relu', return_sequences=True ))
    for i in range(num_lstm_layers):
        name = 'layer_lstm_decoder_{0}'.format(i+1)
        model.add(LSTM(16, activation='relu', return_sequences=True, name=name))
    model.add(LSTM(32, activation='relu', return_sequences=True)) 
    model.add(Dropout(rate=drop_rate_2))
    model.add(TimeDistributed(Dense(n_features)))
 
    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mae',  metrics=['accuracy'])
    model.summary()
    
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
  

def preprocess(raw, timesteps):
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
    
    
def generate_full_X_train(normal_sequence, timesteps):
    X_train_full = list()
    size = len(normal_sequence)
    if size  > 1:
        X_train_full = pd.concat(normal_sequence)
    else:
        print("normal_sequence", normal_sequence)
        print("size", len(normal_sequence))
        print(normal_sequence[0])
        X_train_full = normal_sequence[0]
    print(type(X_train_full))
    stime ="01-01-2017 00:00:00"
    etime ="01-03-2017 00:00:00"

    frmt = '%d-%m-%Y %H:%M:%S'
    min_date = datetime.datetime.strptime(stime, frmt)
    max_date = datetime.datetime.strptime(etime, frmt)
    print("min date", type(min_date))
    print("type", X_train_full.dtypes)
    
    X_train_full = select_data(X_train_full, min_date, max_date)
    X_train_full = preprocess(X_train_full, timesteps)
    return X_train_full

def generate_sets(normal_sequence, timesteps):
    print("normal_sequence", normal_sequence)
    
    stime ="01-01-2017 00:00:00"
    etime ="01-03-2017 00:00:00"

    frmt = '%d-%m-%Y %H:%M:%S'
    min_date = datetime.datetime.strptime(stime, frmt)
    max_date = datetime.datetime.strptime(etime, frmt)
    X_train_D = select_data(normal_sequence, min_date, max_date)
    
    size_X_train_D = X_train_D.shape[0]
    size_train = round(size_X_train_D*0.8)
    
    X_train = X_train_D.iloc[:size_train, :]
    X_val = X_train_D.iloc[size_train:, :]
  
    size_val = round(0.5*X_val.shape[0])
   
    X_val_1_D = X_val.iloc[:size_val, :]
    X_val_2_D = X_val.iloc[size_val:, :]
    
    X_train = preprocess(X_train_D, timesteps)
    X_val_1 = preprocess(X_val_1_D, timesteps)
    X_val_2 = preprocess(X_val_2_D, timesteps)
    
    return X_train, X_val_1, X_val_2

def generate_sets_days(normal_sequence, timesteps):
    print("normal_sequence", type(normal_sequence))
    
    stime ="01-01-2017 00:00:00"
    etime ="01-03-2017 00:00:00"

    frmt = '%d-%m-%Y %H:%M:%S'
    min_date = datetime.datetime.strptime(stime, frmt)
    max_date = datetime.datetime.strptime(etime, frmt)
    
    X_train_D = select_data(normal_sequence, min_date, max_date)
    
    size_X_train_D = X_train_D.shape[0]
    size_train = round(size_X_train_D*0.8)
    
    X_train = X_train_D.iloc[:size_train, :]
    X_val = X_train_D.iloc[size_train:, :]
  
    size_val = round(0.5*X_val.shape[0])
   
    X_val_1_D = X_val.iloc[:size_val, :]
    X_val_2_D = X_val.iloc[size_val:, :]
    
    return X_train, X_val_1_D, X_val_2_D
    
#see https://towardsdatascience.com/lstm-autoencoder-for-anomaly-detection-e1f4f2ee7ccf
#https://towardsdatascience.com/lstm-autoencoder-for-extreme-rare-event-classification-in-keras-ce209a224cfb
def test_autoencoder(bayesian=False):
    global config
    sequence, normal_sequence, anomalous_sequence = generate_sequences("12", "sensortgmeasurepp", limit=True, df_to_csv=True)
    
    if bayesian == True:
        param = do_bayesian_optimization()
        config = param
        
    num_lstm_layers = get_num_layers_lstm(config)
    batch_size = get_dim_batch_size(config)
    learning_rate = get_dim_learning_rate(config)
    drop_rate_1 = get_dim_drop_rate_1(config)
    drop_rate_2 = get_dim_drop_rate_2(config)
    print("number of lstm layers:", num_lstm_layers)
    print("batch size:", batch_size)
    print("learning rate:", learning_rate)
    print("drop rate 1:", drop_rate_1)
    print("drop rate 2:", drop_rate_2)
    
  
        
    clear_session()
    #tensorflow.reset_default_graph()
        
    #1 minute frequency size of sliding window 1440- day
    #it already seems unfeasible
    #week 10080
    #15 min frequency a day is 96
    timesteps = 96
    
    X_train_full = generate_full_X_train(normal_sequence,timesteps)
    model = autoencoder_model(X_train_full, num_lstm_layers, learning_rate, drop_rate_1, drop_rate_2)
    
    number_of_chunks = 0
    history = list()
    is_best_model = False
    for df_chunk in normal_sequence:
        if is_best_model:
            model = load_model("best_autoencoderLSTM.h5")
        number_of_chunks += 1
        print("number of chunks:", number_of_chunks)
        X_train, X_val_1, X_val_2 = generate_sets(df_chunk, timesteps)  
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        mc = ModelCheckpoint('best_autoencoderLSTM.h5', monitor='val_loss', mode='min', save_best_only=True)
        history = model.fit(X_train, X_train, validation_data=(X_val_1, X_val_1), epochs=20, batch_size=batch_size, callbacks=[es, mc]).history
        is_best_model = True
        
    model.save("autoencoderLSTM.h5")
    print("Saved model to disk")
    
    model = load_model('autoencoderLSTM.h5')
    print("Loaded model")
        
    plot_training_losses(history)
    X_pred = model.predict(X_train_full)
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
    sigma = get_sigma(vector, mu)
        
    score = anomaly_score(mu, sigma, vector)
 
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
    
    normal_sequence_full = pd.concat(normal_sequence)
 
    _, _, X_val_2_D = generate_sets_days(normal_sequence_full, timesteps)
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
    
    dates_list = list()
    #positive class is anomaly
    FP = 0
    TP = 0
    FN = 0
    i = 0
    for sc in score:
        if sc > min_th:
             FP += 1
             date = X_val_2_D['date'].iloc[i]
             dates_list.append(date)
        i += 1
    
    #question? division by zero
    fbs = f_beta_score(TP, FP, FN, beta=0.1)
    print("f_beta_score", fbs)
    
    
    #accuracy = model.evaluate(X_test, X_test)[1]
    
    #print("accuracy", accuracy)
    
 
    return True


def get_dim_drop_rate_2(dimensions):
    return dimensions[4]
def get_dim_drop_rate_1(dimensions):
    return dimensions[3]
def get_dim_learning_rate(dimensions):
    return dimensions[2]
def get_dim_batch_size(dimensions):
    return dimensions[1]
def get_num_layers_lstm(dimensions):
    return dimensions[0]


def do_bayesian_optimization():
    dimensions,  default_parameters = hyperparam_opt()
    print("START BAYESIAN OPTIMIZATION")
    es = DeltaYStopper(0.01)
    
    #signal.alarm(120)
    gp_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                n_calls=11,
                                noise= 0.01,
                                n_jobs=-1,
                                x0=default_parameters,
                                callback=es, 
                                random_state=12,
                                acq_func="EIps")
    param = gp_result.x     
    clear_session()
    
    return param 
 

def operation(data, anomaly_threshold):
    return True
    

 
