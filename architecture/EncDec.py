# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:45:57 2020

@author: anama
"""

from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from keras.models import Sequential
from keras.models import Model
from preprocessing.series import create_dataset_as_supervised, create_dataset, generate_sequences, series_to_supervised, select_data, generate_normal, change_format
from preprocessing.series import downsample
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.backend import clear_session
from keras import regularizers
from datetime import datetime
from keras.optimizers import Adam
from architecture.evaluate import f_beta_score
import time
import pickle
import os
from architecture import utils
from architecture import tuning 
from report import image, HtmlFile, tag, Text
from base.Network import Network

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
    #stime = "01-01-2017 00:00:00"
    #etime = "01-12-2017 00:00:00"
    stime = None
    etime = None
        
        
    @classmethod
    def do_train(cls, timesteps=96, simulated = False, bayesian=False, save=True, validation=True):
            network = Network("infraquinta")
            path_report = "F:/manual/Tese/exploratory/wisdom/reports_files/report_models/%s.html" % cls.report_name
    
            EncDec.init_report(path_report)
            mu = cls.mu
            sigma = cls.sigma
            
            stime =cls.stime
            etime = cls.etime
            EncDec.n_steps = timesteps
        
            normal_sequence, test_sequence = generate_sequences("12", "sensortgmeasurepp",start=stime, end=etime, simulated=simulated, df_to_csv=True)
          
            config = cls.config
            if bayesian == True:
                fitness = cls.fitness_func
                dimensions = cls.dimensions
                default_parameters = cls.default_parameters
                param = tuning.do_bayesian_optimization(fitness, dimensions, default_parameters)
                cls.config = param
            
            clear_session()
               
            X_train_full, y_train_full = utils.generate_full(normal_sequence,timesteps, input_form=cls.input_form, output_form=cls.output_form, n_seq=cls.n_seq,n_input=cls.n_input, n_features=cls.n_features)
            model = cls.type_model_func(X_train_full, y_train_full, config)
            batch_size = tuning.get_param(config, cls.toIndex, "batch_size")
            number_of_chunks = 0
            history = list()
            is_best_model = False
        
            if simulated == True:
                validation = False
                
            model = cls.type_model_func(X_train_full, y_train_full, config)
            best_h5_filename = "best_" + cls.h5_file_name + ".h5"
            
            if is_best_model:
                model = load_model(best_h5_filename)
            number_of_chunks += 1
            X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2 = utils.generate_sets(normal_sequence, timesteps,input_form = cls.input_form, output_form = cls.output_form, validation=validation, n_seq=cls.n_seq,n_input=cls.n_input, n_features=cls.n_features)  
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
            mc = ModelCheckpoint(best_h5_filename, monitor='val_loss', mode='min', save_best_only=True)
            if validation:
                history = model.fit(X_train, y_train, validation_data=(X_val_1, y_val_1), epochs=20, batch_size=batch_size, callbacks=[es, mc]).history
            else:
                history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size, callbacks=[es, mc]).history
            is_best_model = True
              
            cls.file.append(Text.Text("Loss last epoch:" + str(history['loss'][-1])))
            cls.file.append(Text.Text("Validation loss last epoch:" + str(history['val_loss'][-1])))
            
            
            
            filename = cls.h5_file_name +'.h5'
            path = os.path.join("..//gui_margarida//gui//assets", filename)
            model.save(path)
            print("Saved model to disk")
            
            model = load_model(path)
            print("Loaded model")
                
            if validation == True:
                encoded = utils.plot_training_losses(history)    
                img = image.Image("Training losses graph", encoded)
                cls.file.append(img)
            
                
            y_pred = model.predict(X_train_full)
            y_pred = utils.process_predict(y_pred)
            y_pred = pd.DataFrame(y_pred)
            scored = pd.DataFrame(index=y_pred.index)
            
            if len(y_train.shape)==3:
                ytrain =  np.squeeze(y_train)
                ytrain = ytrain[:,-1]
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
            print("y_val_2", y_val_2.shape)
            vector = utils.np.squeeze(vector)
            score = utils.anomaly_score(mu, sigma, vector)
            
           
            _, _, X_val_2_D = utils.generate_sets_days(normal_sequence, timesteps)
            
            print("x_val_2_D shape", X_val_2_D.shape)
            print("score", len(score))
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
            
           
            prediction = utils.detect_anomalies(test_sequence, cls.h5_file_name)
            events = network.loadEvents()
            cls.file.append(Text.Text("Number of events year:" + str(len(events))))
            
            
            start = min(test_sequence['date'])
            end = max(test_sequence['date'])
            
            start = change_format(start, '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S')
            end = change_format(end, '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S')
            events = network.select_events(start, end)
            cls.file.append(Text.Text("Number of events during %s to %s: %s" % 
                                      (start, end, len(events))))
            
            TP = 0
            FP = 0
            for date in prediction['date']:
                for event in events:
                  print(type(date))
                  print(type(event.getStart()))
                  date = pd.to_datetime(date)
                  
                  if date >= event.getStart() and date <= event.getEnd():
                      print("match")
                      TP += 1
                  else:
                      FP += 1
            
            cls.file.append(Text.Text("True positive:" + str(TP)))
            cls.file.append(Text.Text("False positive:" + str(FP)))
            EncDec.write_report(path_report)
        
            return True
        
    @classmethod
    def write_report(cls, file_name):
        cls.file.writeToHtml(file_name)
        
    @classmethod
    def init_report(cls, file_name):
        cls.file = HtmlFile.HtmlFile()
        html = tag.Html()
        cls.file.append(html)
        head = tag.Head()
        cls.file.append(head)
        body = tag.Body()
        cls.file.append(body)
        
    
    
    
    def operation(data, anomaly_threshold):
            prediction = utils.detect_anomalies(data, EncDec.h5_file_name)
            return prediction

    def do_test():
        return True