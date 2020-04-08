# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:45:57 2020

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
from architecture import utils
from architecture import tuning 

class EncDec(object):
    config = []
    mu = 0
    sigma = 0
    h5_file_name = ""
    type_model_func = None
    type_model = ""
    fitness_func = None
    dimensions = []
    default_parameters = []
    toIndex = dict()
    n_features = 1
    n_epochs = 0
    n_steps = 96
    n_seq = None
    n_input = None
    
        
        
    @classmethod
    def do_train(cls, timesteps=96, simulated = False, bayesian=False, save=True, validation=True):
            print("do_train")
            print("validation", validation)
            print("function", cls.type_model_func)
            print("to index", cls.toIndex)
            mu = cls.mu
            sigma = cls.sigma
            
            stime ="01-01-2017 00:00:00"
            etime ="01-03-2017 00:00:00"
        
            normal_sequence, _ = generate_sequences("12", "sensortgmeasurepp",start=stime, end=etime, simulated=simulated, df_to_csv=True)
            print("test normal_sequence", normal_sequence[0].shape)
            config = cls.config
            if bayesian == True:
                fitness = cls.fitness_func
                dimensions = cls.dimensions
                default_parameters = cls.default_parameters
                param = tuning.do_bayesian_optimization(fitness, dimensions, default_parameters)
                cls.config = param
            
            clear_session()
               
            print("input_form", cls.input_form)
            X_train_full, y_train_full = utils.generate_full(normal_sequence,timesteps, input_form=cls.input_form, output_form=cls.output_form, n_seq=cls.n_seq,n_input=cls.n_input, n_features=cls.n_features)
            model = cls.type_model_func(X_train_full, y_train_full, config)
            batch_size = tuning.get_param(config, cls.toIndex, "batch_size")
            number_of_chunks = 0
            history = list()
            is_best_model = False
        
            if simulated == True:
                validation = False
                
            print("type_model_func", cls.type_model_func)
            model = cls.type_model_func(X_train_full, y_train_full, config)
            best_h5_filename = "best_" + cls.h5_file_name + ".h5"
            for df_chunk in normal_sequence:
                if is_best_model:
                    model = load_model(best_h5_filename)
                number_of_chunks += 1
                print("number of chunks:", number_of_chunks)
                X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2 = utils.generate_sets(df_chunk, timesteps,input_form = cls.input_form, output_form = cls.output_form, validation=validation, n_seq=cls.n_seq,n_input=cls.n_input, n_features=cls.n_features)  
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
                mc = ModelCheckpoint(best_h5_filename, monitor='val_loss', mode='min', save_best_only=True)
                if validation:
                    history = model.fit(X_train, y_train, validation_data=(X_val_1, y_val_1), epochs=20, batch_size=batch_size, callbacks=[es, mc]).history
                else:
                    history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size, callbacks=[es, mc]).history
                is_best_model = True
              
            filename = cls.h5_file_name +'.h5'
            path = os.path.join("..//gui_margarida//gui//assets", filename)
            model.save(path)
            print("Saved model to disk")
            
            model = load_model(path)
            print("Loaded model")
                
            if validation == True:
                utils.plot_training_losses(history)
                
            y_pred = model.predict(X_train_full)
            y_pred = utils.process_predict(y_pred)
            y_pred = pd.DataFrame(y_pred)
            scored = pd.DataFrame(index=y_pred.index)
            
            if len(y_train.shape)==3:
                ytrain =  np.squeeze(y_train)
                ytrain = ytrain[:,0]
                ytrain = ytrain.reshape(ytrain.shape[0],1)
          
            else:
                ytrain = y_train
                  
            scored['Loss_mae'] = np.mean(np.abs(y_pred-ytrain), axis = 1)
            plt.figure(figsize=(16,9), dpi=80)
            plt.title('Loss Distribution', fontsize=16)
            sns.distplot(scored['Loss_mae'], bins=20, kde=True, color='blue')
            plt.show()
                
            #calculate loss on the validation set to get miu and sigma values
            #should define an entire validation set and not only last set from chunk  
            y_pred = model.predict(X_val_1)
            
            vector = utils.get_error_vector(y_val_1, y_pred)
            vector = np.squeeze(vector)    
            plt.hist(list(vector), bins=20)
            plt.show()
                
            vector = vector.reshape(vector.shape[0], 1)
            print("vector shape", vector.shape)
            print("vector", vector)
                
            mu = utils.get_mu(vector)
            print("mu", mu)
            sigma = utils.get_sigma(vector, mu)
            print("sigma", sigma)
            score = utils.anomaly_score(mu, sigma, vector)
         
            y_pred = model.predict(y_val_2) 
            vector = utils.get_error_vector(y_val_2, y_pred)
            
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
                utils.save_parameters(mu, sigma, timesteps, min_th, EncDec.h5_file_name)
            
            return True
        
    def operation(data, anomaly_threshold):
            prediction = utils.detect_anomalies(data, EncDec.h5_file_name)
            return prediction
