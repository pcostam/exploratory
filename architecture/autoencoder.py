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
import pickle
import os
import utils
import tuning

#PARAMETERS
config = [2, 128, 1e-2, 0.5, 0.5]
mu = 0
sigma = 0
timesteps = 0
min_th = 0

#see https://towardsdatascience.com/lstm-autoencoder-for-anomaly-detection-e1f4f2ee7ccf
#https://towardsdatascience.com/lstm-autoencoder-for-extreme-rare-event-classification-in-keras-ce209a224cfb
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
    X_train_full, _ = utils.generate_full(normal_sequence, 96)
    model = autoencoder_model(X_train_full, num_lstm_layers, learning_rate, drop_rate_1, drop_rate_2)
    
    print("total number of chunks", len(normal_sequence))
    no_chunks = 0
    for df_chunk in normal_sequence:
        no_chunks += 1
        print("number of chunks:", no_chunks)
        X_train,_,  X_val_1, _, X_val_2,_ = utils.generate_sets(df_chunk, 96) 
        es = EarlyStopping(monitor='val_loss', min_delta = 0.01, mode='min', verbose=1)
        hist = model.fit(X_train, X_train, validation_data=(X_val_1, X_val_1), epochs=100, batch_size= batch_size, callbacks=[es])
    
    loss = hist.history['loss'][-1]
    
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    clear_session()
    tensorflow.compat.v1.reset_default_graph()
    
    end = time.perf_counter()
    diff = end - init
    
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


def test_autoencoder(simulated = False, bayesian=False, save=True):
    global config, mu, sigma
    
    stime ="01-01-2017 00:00:00"
    etime ="01-03-2017 00:00:00"

    normal_sequence, _ = generate_sequences("12", "sensortgmeasurepp",start=stime, end=etime, simulated=simulated, df_to_csv=True)
    print("test normal_sequence", normal_sequence[0].shape)
    if bayesian == True:
        param = tuning.do_bayesian_optimization()
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
    
    X_train_full, _ = utils.generate_full(normal_sequence,timesteps)
    model = autoencoder_model(X_train_full, num_lstm_layers, learning_rate, drop_rate_1, drop_rate_2)
    
    number_of_chunks = 0
    history = list()
    is_best_model = False
    validation = True
    if simulated == True:
        validation = False
        
    for df_chunk in normal_sequence:
        if is_best_model:
            model = load_model("best_autoencoderLSTM.h5")
        number_of_chunks += 1
        print("number of chunks:", number_of_chunks)
        X_train, _, X_val_1, _, X_val_2, _ = utils.generate_sets(df_chunk, timesteps, validation=validation)  
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        mc = ModelCheckpoint('best_autoencoderLSTM.h5', monitor='val_loss', mode='min', save_best_only=True)
        if validation:
            history = model.fit(X_train, X_train, validation_data=(X_val_1, X_val_1), epochs=20, batch_size=batch_size, callbacks=[es, mc]).history
        else:
            history = model.fit(X_train, X_train, epochs=20, batch_size=batch_size, callbacks=[es, mc]).history
        is_best_model = True
      
    filename = 'autoencoderLSTM.h5'
    path = os.path.join("..//gui_margarida//gui//assets", filename)
    model.save(path)
    print("Saved model to disk")
    
    model = load_model(path)
    print("Loaded model")
        
    if validation == False:
        utils.plot_training_losses(history)
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
    #should define an entire validation set and not only last set from chunk  
    X_pred = model.predict(X_val_1)
        
    vector = utils.get_error_vector(X_val_1, X_pred)
    vector = np.squeeze(vector)    
    plt.hist(list(vector), bins=20)
    plt.show()
        
    vector = vector.reshape(vector.shape[0], 1)
    print("vector shape", vector.shape)
    print(vector)
        
    mu = utils.get_mu(vector)
    sigma = utils.get_sigma(vector, mu)
        
    score = utils.anomaly_score(mu, sigma, vector)
 
    X_pred = model.predict(X_val_2) 
    vector = utils.get_error_vector(X_val_2, X_pred)
    
    vector = utils.np.squeeze(vector)
    score = utils.anomaly_score(mu, sigma, vector)
    
    normal_sequence_full = pd.concat(normal_sequence)
 
    _, _, X_val_2_D = utils.generate_sets_days(normal_sequence_full, timesteps)
    
    min_th = utils.get_threshold(X_val_2_D, score)

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
    
    if save == True:
        save_parameters(mu, sigma, timesteps, min_th)
    
    return True

def save_parameters(mu, sigma, timesteps, min_th):
    param = {'mu':mu, 'sigma':sigma, 'timesteps':timesteps, 'min_th':min_th}
    filename = 'parametersAutoencoderLSTM.pickle'
    path = os.path.join("..//gui_margarida//gui/assets", filename)
    with open(path, 'wb') as f:  
        pickle.dump(param, f, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def load_parameters():
    global mu, sigma, timesteps, min_th
    filename = 'parametersAutoencoderLSTM.pickle'
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

def detect_anomalies(X_test, choose_th=None):
    print("shape", X_test.shape)
    param = load_parameters()
    
    
    
    filename = 'autoencoderLSTM.h5'
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
    
    Xtest = utils.preprocess(X_test, timesteps)
    print("Xtest shape", Xtest.shape)
    print("Xtest type", type(Xtest))
    
    print("Predict")
    X_pred = model.predict(Xtest)

    predict = pd.DataFrame()
    
    vector = utils.get_error_vector(Xtest, X_pred)
    vector = np.squeeze(vector)
        
    score = utils.anomaly_score(mu, sigma, vector)
    
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

def operation(data, anomaly_threshold):
    prediction = detect_anomalies(data, None)
    return prediction
    

 
