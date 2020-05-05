# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:45:57 2020

@author: anama
"""

from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from keras.models import Sequential
from keras.models import Model
from preprocessing.series import create_dataset_as_supervised, create_dataset, generate_sequences, series_to_supervised, generate_normal, change_format
from preprocessing.series import downsample, rolling_out_cv, generate_total_sequence, select_data, csv_to_df, split_train_test
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
from keras.utils import plot_model
from kstest import goodness_of_fit, ecdf

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
    split = False
        
        
    @classmethod
    def do_train(cls, timesteps=96, cv=True, simulated = False, bayesian=False, save=True, validation=True, hidden_size=16, code_size=4):
            network = Network("infraquinta")
            path_report = "F:/manual/Tese/exploratory/wisdom/reports_files/report_models/%s.html" % cls.report_name
    
            EncDec.init_report(path_report)
            mu = cls.mu
            sigma = cls.sigma
            stime =cls.stime
            etime = cls.etime
            EncDec.n_steps = timesteps
            cls.hidden_size = hidden_size
            cls.code_size = code_size
            
            
            cls.file.append(Text.Text("Hidden size:" + str(cls.hidden_size)))
            cls.file.append(Text.Text("Code size:" + str(cls.code_size)))
            cls.file.append(Text.Text("Dropout:" + str(cls.dropout)))
            cls.file.append(Text.Text("Regularization:" + str(cls.regularizer)))
            cls.file.append(Text.Text("Timesteps:" + str(timesteps)))
           
        
            normal_sequence, test_sequence = generate_sequences("12", "sensortgmeasurepp",start=stime, end=etime, simulated=simulated, df_to_csv=True)
            normal_sequence_2, test_sequence_2 = generate_sequences("11", "sensortgmeasurepp", start=stime, end=etime, simulated=simulated, df_to_csv=True)
           
            sequence = generate_total_sequence("12", "sensortgmeasurepp",start=stime, end=etime)
            sequence_2 = generate_total_sequence("11", "sensortgmeasurepp",start=stime, end=etime)
            
            config = cls.config
            if bayesian == True:
                fitness = cls.fitness_func
                dimensions = cls.dimensions
                default_parameters = cls.default_parameters
                print("EncDec dimensions", EncDec.dimensions)
                param = tuning.do_bayesian_optimization(fitness, dimensions, default_parameters)
                cls.config = param
            
            clear_session()
            
            train_chunks, test_chunks = list(), list()
            train_chunks_all, test_chunks_all = list(), list()
            #3 MESES
            n_train = 8640
            #cross validation
            if cv:
                train_chunks, test_chunks = split_train_test(normal_sequence, sequence, n_train)
                train_chunks_all.append(train_chunks)
                test_chunks_all.append(test_chunks)
                train_chunks, test_chunks = split_train_test(normal_sequence_2, sequence, n_train)
                train_chunks_all.append(train_chunks)
                test_chunks_all.append(test_chunks)
            
            
            k = len(train_chunks)
            print("number of partitions", k)
            cls.file.append(Text.Text("Number of partitions:" + str(k)))
            
            train_chunks = list()
            for item in train_chunks_all:
                print("item:", type(item[0]))
                
            for i in range(0, k):
                    to_concat = [item[i] for item in train_chunks_all] 
                    union_normal_sequences = pd.concat(to_concat, axis=1) 
                    union_normal_sequences =  union_normal_sequences.drop_duplicates("date", "first")
                    col = (union_normal_sequences.columns == 'date').argmax()
                    dates = union_normal_sequences.iloc[:, col]
                    union_normal_sequences = union_normal_sequences.drop(['date'], axis=1)
                    union_normal_sequences['date'] = dates                 
                    train_chunks.append(union_normal_sequences)
            
            EncDec.n_features = len(train_chunks[0].columns) - 1
            print("TRAIN_CHUNKS", train_chunks[0].columns)
            run_losses = list()
            run_val_losses = list()
            for i in range(k):  
       
                normal_sequence = train_chunks[i]
                print("normal_sequence dates", train_chunks[i]['date'])
                test_sequence = test_chunks[i]
                size_train = len(normal_sequence)
                print("Training size:", size_train)
                cls.file.append(Text.Text("Training size:" + str(size_train)))
                start_train = min(normal_sequence['date'])
                end_train = max(normal_sequence['date'])
                cls.file.append(Text.Text("Date train events %s to %s" % 
                                              (start_train, end_train)))
            
               
               
                X_train_full, y_train_full = utils.generate_full(normal_sequence,timesteps, input_form=cls.input_form, output_form=cls.output_form, n_seq=cls.n_seq,n_input=cls.n_input, n_features=cls.n_features)
              
                model = cls.type_model_func(X_train_full, y_train_full, config)
                batch_size = tuning.get_param(config, cls.toIndex, "batch_size")
                number_of_chunks = 0
                history = list()
                is_best_model = False
            
                if simulated == True:
                    validation = False
                    
                best_h5_filename = "best_" + cls.h5_file_name + ".h5"
                
                if is_best_model:
                    model = load_model(best_h5_filename)
                number_of_chunks += 1
                X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2 = utils.generate_sets(normal_sequence, timesteps,input_form = cls.input_form, output_form = cls.output_form, validation=validation, n_seq=cls.n_seq,n_input=cls.n_input, n_features=cls.n_features)  
                es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=5, verbose=1)
                mc = ModelCheckpoint(best_h5_filename, monitor='val_loss', mode='min', save_best_only=True)
                
             
                if validation:
                    if EncDec.split:
                        X_train = utils.split_features(EncDec.n_features, X_train)
                        X_val_1 = utils.split_features(EncDec.n_features, X_val_1)
                        
                    history = model.fit(X_train, y_train, validation_data=(X_val_1, y_val_1), epochs=200, batch_size=batch_size, callbacks=[es, mc]).history
                else:
                    history = model.fit(X_train, y_train, epochs=200, batch_size=batch_size, callbacks=[es, mc]).history
                is_best_model = True
                
                plot_model(model, to_file='model.png')
                
                loss = history['loss'][-1]
                val_loss = history['val_loss'][-1]
                cls.file.append(Text.Text("Loss last epoch:" + str(loss)))
                cls.file.append(Text.Text("Validation loss last epoch:" + str(val_loss)))
                run_losses.append(loss)
                run_val_losses.append(val_loss)
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
                ytrain = utils.process_predict(y_train_full)
                scored['Loss_mae'] = np.mean(np.abs(y_pred-ytrain), axis = 1)
                scored['Loss_ae'] = np.abs(y_pred-ytrain)
                print("loss ae shape", scored['Loss_ae'].shape)
                print("loss mae shape", scored['Loss_mae'].shape)
                encoded = utils.plot_bins_loss(scored['Loss_ae'])
                img = image.Image("Training losses graph", encoded)
                cls.file.append(img)
                
                goodness_of_fit(cls.file, scored['Loss_ae'], alpha=0.05)
                img = ecdf(scored['Loss_ae'])
                img = image.Image("Cumulative Density Function", encoded)
                cls.file.append(img)
                
                
                #calculate loss on the validation set to get miu and sigma values
                y_pred = model.predict(X_val_1)
   
                vector = utils.get_error_vector(y_val_1, y_pred)
                vector = np.squeeze(vector)    
                plt.hist(list(vector), bins=20)
                plt.show()
                    
                vector = vector.reshape(vector.shape[0], 1)
                
                X_train_D, X_val_1_D, X_val_2_D = utils.generate_sets_days(normal_sequence, cls.n_input)
                print("X_train shape", X_train.shape)
                print("X_train_D", X_train_D.shape)
                print("n_input", cls.n_input)
               
                dates = X_val_1_D['date']
                
                title = "Time series validation set 1"
                encoded = utils.plot_series(title, dates, y_val_1, y_pred)
                img = image.Image(title, encoded)
                cls.file.append(img)
                
                    
                mu = utils.get_mu(vector)
            
                sigma = utils.get_sigma(vector, mu)
              
                score = utils.anomaly_score(mu, sigma, vector)
                
                dates = X_val_2_D['date']
                y_pred = model.predict(X_val_2) 
                title = "Time series validation set 2"
                print("dates", len(dates))
                print("y_val_2 shape", y_val_2.shape)
                print("y_pred", y_pred.shape)
                encoded = utils.plot_series(title, dates, y_val_2, y_pred)
                img = image.Image(title, encoded)
                cls.file.append(img)
                
                vector = utils.get_error_vector(y_val_2, y_pred)
             
                vector = utils.np.squeeze(vector)
                score = utils.anomaly_score(mu, sigma, vector)
                
               
                
    
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
                
                
                if save == True:
                    utils.save_parameters(mu, sigma, timesteps, min_th, EncDec.h5_file_name)
                 
                X_test, y_test, _, _, _, _ = utils.generate_sets(test_sequence, timesteps,input_form =cls.input_form, output_form = cls.output_form, validation=False, n_seq=cls.n_seq, n_input=cls.n_input, n_features=cls.n_features)
                
                X_test_D = utils.generate_sets_days(test_sequence, cls.n_input, validation=False)
                dates = X_test_D['date']
                print("dates", len(dates))
                y_pred = model.predict(X_test)
                title = "Time series test sequence"
                encoded = utils.plot_series(title, dates, y_test, y_pred)
                img = image.Image(title, encoded)
                cls.file.append(img)
                
                y_pred = model.predict(X_test)
                prediction = utils.detect_anomalies(X_test, y_test, test_sequence, cls.h5_file_name)
                events = network.loadEvents()
                cls.file.append(Text.Text("Number of events year:" + str(len(events))))
                
                
                old_start = min(test_sequence['date'])
                old_end = max(test_sequence['date'])
                
                start = change_format(old_start, '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S')
                end = change_format(old_end, '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S')
                events = network.select_events(start, end)
                cls.file.append(Text.Text("Number of events during %s to %s: %s" % 
                                          (start, end, len(events))))
                
               
                TP = 0
                FP = 0
                FN = 0
                TN = 0
                detected_events = dict()
                for date in prediction['date']:
                    for event in events:
                      date = pd.to_datetime(date)
                      if date >= event.getStart() and date <= event.getEnd():
                          detected_events[event.getId()] = event
                          print("match")
                          TP += 1
                      else:
                          FP += 1
                
                path = "F:\\manual\\Tese\\exploratory\\wisdom\\dataset\\infraquinta\\real\\mask\\sensor_"+ str(12) + ".csv"
                df = csv_to_df(12, path, limit = True, n_limit=1000)
                df = select_data(df, old_start, old_end)
                
                for index, row in df.iterrows():
                    if row['anomaly'] == 1 and row['date'] not in prediction['date']:
                        FN += 1
                    if row['anomaly'] == 0 and row['date'] not in prediction['date']:
                        TN += 1
                
                no_detected_events = len(list(detected_events.keys()))
                cls.file.append(Text.Text("Number detected events:" + str(no_detected_events)))
                cls.file.append(Text.Text("True positive:" + str(TP)))
                cls.file.append(Text.Text("False positive:" + str(FP)))
                cls.file.append(Text.Text("False negative:" + str(FN)))
                cls.file.append(Text.Text("True negative:" + str(TN)))
            
            mean_loss = np.mean(np.array(run_losses))
            mean_val_loss = np.mean(np.array(run_val_losses))
            cls.file.append(Text.Text("Mean loss:" + str(mean_loss)))
            cls.file.append(Text.Text("Mean validation loss:" + str(mean_val_loss)))
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