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
import pickle
import os
from architecture import utils
from architecture import tuning 
from report import image, HtmlFile, tag, Text
from base.Network import Network
from keras.utils import plot_model
from kstest import goodness_of_fit, ecdf
from architecture.timeseries import timeseries
from architecture.parameters import parameters

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
    n_epochs = 0
    #stime = "01-01-2017 00:00:00"
    #etime = "01-12-2017 00:00:00"
    stime = None
    etime = None
    split = False
    toIndex = dict()
    all_timeseries = []
    validation = True
    parameters = parameters()
    def add_timeseries(timeseries):
        EncDec.all_timeseries.append(timeseries)
        
    def generate_sequences_sensors(sensors, time, simulated, network, stime=None, etime=None):
        """
        Parameters
        ----------
        sensors : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        all_normal_sequence = list()
        all_sequence = list()
        for sensor_id in sensors:
                normal_sequence, _ = generate_sequences(sensor_id, "sensortgmeasurepp",column_time=time, simulated=simulated, df_to_csv=True, no_leaks=network.getNoLeaks())
                all_normal_sequence.append(normal_sequence)
                sequence = generate_total_sequence(sensor_id, "sensortgmeasurepp",start=stime, end=etime, simulated=simulated, no_leaks=network.getNoLeaks())
                all_sequence.append(sequence)
        return all_sequence, all_normal_sequence
    
    def split_train_test_dataframe(sensors, cv, n_train, all_normal_sequence, all_sequence, time):
        """
        

        Parameters
        ----------
        sensors : TYPE
            DESCRIPTION.

        Returns
        -------
        train_chunks_all : TYPE
            DESCRIPTION.
        test_chunks_all : TYPE
            DESCRIPTION.

        """
        train_chunks, test_chunks = list(), list()
        train_chunks_all, test_chunks_all = list(), list()
            
            
        if cv:
            for i in range(len(sensors)):
                train_chunks, test_chunks = split_train_test(all_normal_sequence[i], all_sequence[i], n_train, time=time)
                train_chunks_all.append(train_chunks)
                test_chunks_all.append(test_chunks)
        return train_chunks_all, test_chunks_all
    
    def verifySizes(sequence, n_input, n_steps, n_seq, text):
         if len(sequence) < n_input:
            raise ValueError("""Size input is too big for""" + text + """sequence (size input: %s, timesteps: %s, number of sequences: %s). 
                             Please, try another configuration of timesteps 
                             and number of sequences. Note that n_input=timesteps*n_seq should be less than
                             %s
                             """ % (n_input, EncDec.parameters.get_n_steps(), n_seq, len(sequence)))
      
    def positives(anomalies, events,time):
        detected_events = dict()
        columns_normal = []
        columns_event = []
        TP = 0
        FP = 0
        i = 0
        for date in anomalies[time]:
              for event in events:
                if time == 'date':
                    date = pd.to_datetime(date)
                if date >= event.getStart() and date <= event.getEnd():
                    detected_events[event.getId()] = event
                    columns_event.append(anomalies.iloc[i,:])
                    TP += 1
                else:
                    columns_normal.append(anomalies.iloc[i,:])
                    FP += 1
              i += 1
        return TP, FP, columns_event, columns_normal, detected_events
    
    def negatives(columns_event, columns_normal, test_sequence, events, time):
        FN = 0
        TN = 0   
        snormal_pred = pd.DataFrame()
        if columns_normal != []:
             snormal_pred = pd.concat(columns_normal, names=['value', 'time'])
         
        sevents_true = Network.findSubSequenceWithEvent(test_sequence, events, time)
        snormal_true = Network.findSubSequenceNoEvent(test_sequence, events, time)
       
        if not(sevents_true.empty) and not(snormal_pred.empty):
             FN = pd.merge(sevents_true, snormal_pred, how='inner', on=[time]).shape[0]
        if not(snormal_true.empty) and (snormal_pred.empty):
             TN = pd.merge(snormal_true, snormal_pred, how='inner', on=[time]).shape[0]
        return FN, TN
    
    def confusion_matrix(anomalies, events, time, test_sequence):
           TP, FP, columns_event, columns_normal, detected_events = EncDec.positives(anomalies, events,time)
           FN, TN = EncDec.negatives(columns_event, columns_normal, test_sequence, events, time)
      
           return {'TP':TP, 'FP':FP, 'FN':FN, 'TN':TN, 'detected_events':detected_events}
    
    def getTP(matrix):
        return matrix['TP']
    
    def getTN(matrix):
        return matrix['TN']
    
    def getFN(matrix):
        return matrix['FN']
    
    def getFP(matrix):
        return matrix['FP']
    def get_losses(history):
        loss = history['loss'][-1]
        val_loss = history['val_loss'][-1]
        return loss, val_loss
  
    def get_anomaly_scores():
        return True
    

    
    def do_f_beta_score(min_th, score, dates):
        dates_list = list()
        #positive class is anomaly
        FP = 0
        TP = 0
        FN = 0
        index_date = 0    
        print("len score", score.shape)
        for sc in score:
            if sc > min_th:
                 FP += 1
                 print("index", index_date)
                 print("size dates", dates.shape)
                 date = dates.iloc[index_date]
                 dates_list.append(date)
        index_date += 1
        
        #question? division by zero
        fbs = f_beta_score(TP, FP, FN, beta=0.1)
        print("f_beta_score", fbs)
        return fbs
    def plot_all_time_series():
        for ts in EncDec.all_timeseries:
            encoded = utils.plot_series(ts.get_name(), ts.get_dates(), ts.get_y_true(), ts.get_y_pred(), EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features())
            img = image.Image(ts.get_name(), encoded)
            EncDec.file.append(img)
    
    def training_losses(model, X_train_full, y_train_full):
        y_pred = model.predict(X_train_full)
        y_pred = utils.process_predict(y_pred,  EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features())
        y_pred = pd.DataFrame(y_pred)
        scored = pd.DataFrame(index=y_pred.index)
        ytrain = utils.process_predict(y_train_full,  EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features())
        struct = {'y_pred': y_pred, 'ytrain': ytrain}
        dbfile = open('struct', 'ab')   
        # source, destination 
        pickle.dump(struct, dbfile)                      
        dbfile.close() 
        scored['Loss_mae'] = np.mean(np.abs(y_pred-ytrain), axis = 1)
        #scored['Loss_ae'] = np.abs(y_pred-ytrain)
        #print("loss ae shape", scored['Loss_ae'].shape)
        print("loss mae shape", scored['Loss_mae'].shape)
        encoded = utils.plot_bins_loss(scored['Loss_mae'])
        img = image.Image("Training losses graph", encoded)
        EncDec.file.append(img)
                
        goodness_of_fit(EncDec.file, scored['Loss_mae'], alpha=0.05)
        img = ecdf(scored['Loss_mae'])
        img = image.Image("Cumulative Density Function", encoded)
        EncDec.file.append(img)
   
    
    def save_model(model, h5_file_name):
        filename = h5_file_name +'.h5'
        path = os.path.join("..//gui_margarida//gui//assets", filename)
        model.save(path)
    
        print("Saved model to disk")
        return model
        
    def load_model(path):
        model = load_model(path)
        print("Loaded model")
        return model
    def predict_error(model, X_val, y_val, dates, title):
        y_pred = model.predict(X_val)
        vector = utils.get_error_vector(y_val, y_pred, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features())
        for n in range(EncDec.parameters.get_n_features()):
            ts = timeseries(title, dates, y_val[:,n], y_pred[:,n])
            EncDec.add_timeseries(ts) 
        return vector, y_pred
        
        
    def test_stats(X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2, X_train_full, y_train_full, model, network, test_sequence,  normal_sequence, time, history):
        loss, val_loss = EncDec.get_losses(history)
      
        dates_train, dates_val_1, dates_val_2 = utils.generate_days(normal_sequence, EncDec.parameters.get_n_input(), time)
        n_features = EncDec.parameters.get_n_features()
        EncDec.file.append(Text.Text("Loss last epoch:" + str(loss)))
        EncDec.file.append(Text.Text("Validation loss last epoch:" + str(val_loss)))
                 
        if EncDec.validation == True:
            encoded = utils.plot_training_losses(history)    
            img = image.Image("Training losses graph", encoded)
            EncDec.file.append(img)
            
        #training losses
        EncDec.training_losses(model, X_train_full, y_train_full)
     
        #calculate loss on the validation set to get miu and sigma values on validation set 1
        vector, _ = EncDec.predict_error(model, X_val_1, y_val_1, dates_val_1, "Time series validation set 1")
        
        mu = utils.get_mu(vector)
        sigma = utils.get_sigma(vector, mu) 
        timeseries.do_histogram(vector)
        score = utils.anomaly_score(mu, sigma, vector, n_features)
        
        #calculate new error vector for validation set 2 and get threshold
        vector, _ = EncDec.predict_error(model, X_val_2, y_val_2, dates_val_2, "Time series validation set 2")
        
        score = utils.anomaly_score(mu, sigma, vector, n_features)
       
        min_th = utils.get_threshold(dates_val_2, score)
        
        #sanity check
        EncDec.do_f_beta_score(min_th, score, dates_val_2)
       
        if EncDec.save == True:
           utils.save_parameters(mu, sigma, EncDec.parameters.get_n_steps(), min_th, EncDec.h5_file_name)
       
        print("test_sequence", test_sequence.shape)
        X_test, y_test, _, _, _, _ = utils.generate_sets(test_sequence, EncDec.parameters.get_n_steps(),input_form =EncDec.input_form, output_form = EncDec.output_form, validation=False, n_seq=EncDec.parameters.get_n_seq(), n_input=EncDec.parameters.get_n_input(), n_features=EncDec.parameters.get_n_features())
       
    
        test_dates = utils.generate_days(test_sequence, EncDec.parameters.get_n_input(), time, validation=False)
        vector, y_pred = EncDec.predict_error(model, X_test, y_test, test_dates, "Time series test sequence")
          
        #plot all time series
        EncDec.plot_all_time_series()
      
        anomalies = utils.detect_anomalies(X_test, y_test, test_sequence, EncDec.h5_file_name, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), time=time)
        print("getevents")
        events = network.getEvents()
        print("number of events", len(events))
          
        EncDec.file.append(Text.Text("Number of events year:" + str(len(events))))
       
        print("maxmin")
        start = min(test_sequence[time])
        end = max(test_sequence[time])
       
        if time=='date':
           start = change_format(start, '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S')
           end = change_format(end, '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S')
       
        print("break 2")
        print("start", start)
        print("end", end)
        events = network.select_events(start, end,time)
        print("number of events", len(events))
        EncDec.file.append(Text.Text("Number of events during %s to %s: %s" % 
                                 (start, end, len(events))))
        print("break 3")
        matrix = EncDec.confusion_matrix(anomalies, events, time, test_sequence)
        TP = EncDec.getTP(matrix)
        FP = EncDec.getFP(matrix)
        FN = EncDec.getFN(matrix)
        TN = EncDec.getTN(matrix)
        detected_events = matrix['detected_events']
       
        no_detected_events = len(list(detected_events.keys()))
        EncDec.file.append(Text.Text("Number detected events:" + str(no_detected_events)))
        EncDec.file.append(Text.Text("True positive:" + str(TP)))
        EncDec.file.append(Text.Text("False positive:" + str(FP)))
        EncDec.file.append(Text.Text("False negative:" + str(FN)))
        EncDec.file.append(Text.Text("True negative:" + str(TN)))
        return loss, val_loss
    
    def train(network, k, config, file, train_chunks, test_chunks,time, type_model_func, toIndex, validation):
         h5_file_name = EncDec.h5_file_name
         n_features = EncDec.parameters.get_n_features()
         n_input = EncDec.parameters.get_n_input()
         n_steps = EncDec.parameters.get_n_steps()
         n_seq = EncDec.parameters.get_n_seq()
         input_form = EncDec.input_form
         output_form = EncDec.output_form

         
         run_losses = list()
         run_val_losses = list()
         for i in range(k):  
                normal_sequence = train_chunks[i]
                test_sequence = test_chunks[i]
                EncDec.verifySizes(normal_sequence, n_input, n_steps, n_seq, "train")
                EncDec.verifySizes(test_sequence, n_input, n_steps, n_seq, "test")
                    
                print("chunk test", len(test_sequence))
                
                size_train = len(normal_sequence)
                print("Training size:", size_train)
                EncDec.file.append(Text.Text("Training size:" + str(size_train)))
                start_train = min(normal_sequence[time])
                end_train = max(normal_sequence[time])
                EncDec.file.append(Text.Text("Date train events %s to %s" % 
                                              (start_train, end_train)))
            
 
                X_train_full, y_train_full = utils.generate_full(normal_sequence, n_steps, input_form=input_form, output_form=output_form, n_seq=n_seq,n_input=n_input, n_features=n_features)
                print("X_train_full shape", X_train_full.shape)
                print("y_train_full shape", y_train_full.shape)
                print("config", config)
                print("kernel size", config[2])
            
                model = type_model_func(X_train_full, y_train_full, config)
                batch_size = tuning.get_param(config, toIndex, "batch_size")
          
                history = list()
           
                best_h5_filename = "best_" + h5_file_name + ".h5"    
                #model = load_model(best_h5_filename)
                X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2  = utils.generate_sets(normal_sequence, n_steps,input_form = input_form, output_form = output_form, validation=validation, n_seq=n_seq,n_input=n_input, n_features=n_features)  
                print("X_train shape", X_train.shape)
                print("y_train shape", y_train.shape)
                print("X_val_1 shape", X_val_1.shape)
                print("y_val_1 shape", y_val_1.shape)
                print("X_val_2 shape", X_val_2.shape)
                print("y_val_2 shape", y_val_2.shape)
                
                
                
                print("time", time)
                dates_train, dates_val_1, dates_val_2 = utils.generate_days(normal_sequence, n_input, time)
               
                es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=5, verbose=1)
                mc = ModelCheckpoint(best_h5_filename, monitor='val_loss', mode='min', save_best_only=True)
                
             
                if validation:
                    if EncDec.split:
                        X_train = utils.split_features(EncDec.parameters.get_n_features(), X_train)
                        X_val_1 = utils.split_features(EncDec.parameters.get_n_features(), X_val_1)
                        
                    history = model.fit(X_train, y_train, validation_data=(X_val_1, y_val_1), epochs=200, batch_size=batch_size, callbacks=[es, mc]).history
                else:
                    history = model.fit(X_train, y_train, epochs=200, batch_size=batch_size, callbacks=[es, mc]).history
               
                plot_model(model, to_file='model.png')
                filename = EncDec.h5_file_name +'.h5'
                print("filename:", filename)
                path = os.path.join("..//gui_margarida//gui//assets", filename)
                model.save(path)
                print("Saved model to disk")
                model = load_model(path)
                loss, val_loss = EncDec.test_stats(X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2, X_train_full, y_train_full, model, network, test_sequence, normal_sequence, time, history)
                run_losses.append(loss)
                run_val_losses.append(val_loss)
                
               
                
         return run_losses, run_val_losses
    
    def do_means(run_losses, run_val_losses):
        mean_loss = np.mean(np.array(run_losses))
        mean_val_loss = np.mean(np.array(run_val_losses))
        return mean_loss, mean_val_loss
    
    def get_time_column(simulated):
        time = None
        if simulated:
                time = 'time'
        else:
                time = 'date'
        return time
    
    def get_type_data(simulated):
         typeData = 'real'
         if simulated:
            typeData='simulated'
         return typeData
     
    def add_stat(stat_name, stat):
         EncDec.file.append(Text.Text(stat_name + ':' + str(stat)))
         
         
    def add_multiple_stats(map_stat):
        for stat_name, stat in map_stat.items():
            EncDec.add_stat(stat_name, stat)
     
    
    def do_map_stat(hidden_size, code_size, dropout, regularizer, timesteps):
        map_stat = {'Hidden size': hidden_size, 'Code size': code_size,
                    'Dropout': dropout, 'Regularization': regularizer,
                    'Timesteps': timesteps}
        return map_stat
    
    def get_number_of_cv_partitions(train_chunks_all):
        return len(train_chunks_all[0])
    
 
    def get_columns_to_exclude(simulated):
        to_exclude = []
        if simulated:
            to_exclude = ['time', 'leak']
        else:
            to_exclude = ['date']
        return to_exclude
        
    @classmethod
    def do_train(cls, sensors=["12"], iterative=False, timesteps=96, cv=True, simulated = False, bayesian=False, save=True, validation=True, hidden_size=16, code_size=4):
            typeData = EncDec.get_type_data(simulated)
            network = Network("infraquinta", typeData=typeData, chosen_sensors=sensors,
                              no_leaks=1000, load=False)
            time = EncDec.get_time_column(simulated)
        
            path_report = "F:/manual/Tese/exploratory/wisdom/reports_files/report_models/%s.html" % cls.report_name
            EncDec.init_report(path_report)
            cls.parameters.set_n_steps(timesteps)
           
            EncDec.save = save
            n_train = 8640
            #n_train = 700
            EncDec.h5_file_name = cls.h5_file_name
            EncDec.validation = validation
            toIndex = cls.toIndex
            EncDec.file = cls.file
        
            type_model_func = cls.type_model_func
            cls.hidden_size = hidden_size
            cls.code_size = code_size 
            config = cls.config
            to_exclude = EncDec.get_columns_to_exclude(simulated)
            map_stat = EncDec.do_map_stat(cls.hidden_size, cls.code_size, 
                                   cls.parameters.get_dropout(), cls.parameters.get_regularizer(), timesteps)
            
            EncDec.input_form=cls.input_form
            EncDec.output_form=cls.output_form

            file = EncDec.add_multiple_stats(map_stat)
            all_sequence, all_normal_sequence = EncDec.generate_sequences_sensors(sensors, time, simulated, network)
         
            if bayesian == True:
                cls.config = tuning.do_bayesian_optimization(cls.fitness, cls.dimensions, cls.default_parameters)
            
          
            clear_session()   
        
            train_chunks_all, test_chunks_all = EncDec.split_train_test_dataframe(sensors, cv, n_train, all_normal_sequence, all_sequence, time)
            k = EncDec.get_number_of_cv_partitions(train_chunks_all)
            EncDec.add_stat('Number of partitions:', k)
       
            train_chunks = utils.join_partitions_features(train_chunks_all, k, to_exclude)
            test_chunks = utils.join_partitions_features(test_chunks_all, k, to_exclude)
     
            cls.parameters.set_n_features(len(train_chunks[0].columns) - len(to_exclude))
            EncDec.parameters = cls.parameters
            
            run_losses, run_val_losses = EncDec.train(network, k, config, file, train_chunks, test_chunks,time, type_model_func, toIndex, validation)
            mean_loss, mean_val_loss = EncDec.do_means(run_losses, run_val_losses)
            
            map_stat = {'Mean loss': mean_loss, 'Mean validation loss': mean_val_loss}
            EncDec.add_multiple_stats(map_stat)
            EncDec.write_report(path_report)
            
            return True
        
    def write_report(file_name):
        EncDec.file.writeToHtml(file_name)
        
  
    def init_report(file_name):
        EncDec.file = HtmlFile.HtmlFile()
        html = tag.Html()
        EncDec.file.append(html)
        head = tag.Head()
        EncDec.file.append(head)
        body = tag.Body()
        EncDec.file.append(body)
        
    
    
    
    def operation(data, anomaly_threshold):
            prediction = utils.detect_anomalies(data, EncDec.h5_file_name)
            return prediction

    def do_test():
        return True