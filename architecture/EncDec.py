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
from preprocessing.splits import expanding_window
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
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
from preprocessing.seasonality import inverse_difference
from architecture.corrective_model import temperature_correction_factor
from evaluation.metrics import confusion_matrix, getTP, getFP, getFN, getTN, make_metrics
from evaluation.bootstraping import test_blocks
from anomaly_score import get_threshold_tailed, get_threshold_f_beta_score, test_betas
import random as rn
import tensorflow as tf
from keras import backend as K
class EncDec(object):
    config = []
    mu = 0
    sigma = 0
    min_th = 0
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
    normal_sequence = pd.DataFrame()
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
                normal_sequence, _ = generate_sequences(sensor_id, "sensortgmeasurepp",column_time=time, simulated=simulated, df_to_csv=True, no_leaks=network.getNoLeaks(), start=stime, end=etime)
                print("normal sequence index", normal_sequence.index)
                all_normal_sequence.append(normal_sequence)
                sequence = generate_total_sequence(sensor_id, "sensortgmeasurepp",start=stime, end=etime, simulated=simulated, no_leaks=network.getNoLeaks())
                all_sequence.append(sequence)
        #column-wise   
        df_union_sequence = pd.concat(all_sequence, axis=1)   
        df_union_normal = pd.concat(all_normal_sequence, axis=1)  
        
        print("df union index", df_union_sequence.index)
        return df_union_sequence, df_union_normal
    
    def split_train_test_dataframe(sensors, cv, n_train, all_normal_sequence, all_sequence, time, events):
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
                        
        if cv:
                train_chunks, test_chunks = expanding_window(all_normal_sequence, all_sequence, n_train, events, time=time)
        
        print("train_chunks", len(train_chunks))
        print("test_chunks", len(test_chunks))
    
        
        return train_chunks, test_chunks
    
    def verifySizes(sequence, n_input, n_steps, n_seq, text):
         if len(sequence) < n_input:      
                raise ValueError("""Size input is too big for""" + text + """sequence (size input: %s, timesteps: %s, number of sequences: %s). 
                             Please, try another configuration of timesteps 
                             and number of sequences. Note that n_input=timesteps*n_seq should be less than
                             %s
                             """ % (n_input, EncDec.parameters.get_n_steps(), n_seq, len(sequence)))
     
      
   
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
                 date = dates[index_date]
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
    def graph_components(map_id_avg, threshold):
        from evaluation import graphs
        encoded = graphs.bars_leaks(map_id_avg, threshold)
        img = image.Image("Mean Anomaly Score by Leak", encoded)
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
        encoded = utils.plot_bins_loss('Loss Distribution', scored['Loss_mae'])
        img = image.Image("Training losses graph", encoded)
        EncDec.file.append(img)
                
        goodness_of_fit(EncDec.file, "Loss Mean absolute error", scored['Loss_mae'], alpha=0.05)
        img = ecdf(scored['Loss_mae'])
        img = image.Image("Cumulative Density Function", encoded)
        EncDec.file.append(img)
   
    
    def save_model(model, h5_file_name):
        filename = h5_file_name +'.h5'
        path = os.path.join(os.getcwd(),"wisdom/gui_margarida/gui/assets", filename)
        model.save(path)
    
        print("Saved model to disk")
        return model
        
    def load_model(path):
        model = load_model(path)
        print("Loaded model")
        return model
    
    def get_y_true(sensorId,start=None, end=None, column_time='date',  simulated=False, mode='normal'):
         if mode == 'normal':
               print("normal")
               print("start", start)
               print("end", end)
               data = generate_normal(sensorId, start=start, end=end, column_time=column_time, simulated=simulated,adjustment=False)
         else:   
               data = generate_total_sequence(sensorId, "sensortgmeasurepp", start, end, simulated=simulated)
         print("data normal", len(data))
         return data
     
    def predict_error(scaler_normal, model, X_val, y_val, y_inv, dates, title, mode="normal"):
        y_pred = model.predict(X_val)
        vector = utils.get_error_vector(y_val, y_pred, y_inv, scaler_normal, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features())
        #temperature_correction_factor(vector, dates, 'date')
      
        for n in range(EncDec.parameters.get_n_features()):
            ts = timeseries(title, dates, y_val[:,n], y_pred[:,n])
            EncDec.add_timeseries(ts) 
        return vector, y_pred
        
   
    def test_stats(scaler_normal, y_val_1_inv, y_val_2_inv, y_anormal_inv, X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2, X_train_full, y_train_full, model, network, anormal_sequence,  normal_sequence, time, history):
        loss, val_loss = EncDec.get_losses(history)
      
        dates_train, dates_val_1, dates_val_2 = utils.generate_days(normal_sequence, EncDec.parameters.get_n_input(), time)
        n_features = EncDec.parameters.get_n_features()
        EncDec.file.append(Text.Text("Loss last epoch:" + str(loss)))
        EncDec.file.append(Text.Text("Validation loss last epoch:" + str(val_loss)))
                 
        if EncDec.validation == True:
            encoded = utils.plot_training_losses(history)    
          
        #training losses
        EncDec.training_losses(model, X_train_full, y_train_full)
     
        #calculate loss on the validation set 1 to get miu and sigma values on validation set 1
        vector, _ = EncDec.predict_error(scaler_normal, model, X_val_1, y_val_1, y_val_1_inv, dates_val_1, "Time series validation set 1")
        
        mu = utils.get_mu(vector)
        EncDec.mu = mu
        sigma = utils.get_sigma(vector, mu) 
        EncDec.sigma = sigma
        timeseries.do_histogram(vector)
        score = utils.anomaly_score(mu, sigma, vector, n_features)
        
        #calculate new error vector for validation set 2 
        vector, _ = EncDec.predict_error(scaler_normal, model, X_val_2, y_val_2, y_val_2_inv, dates_val_2, "Time series validation set 2")        
        score = utils.anomaly_score(mu, sigma, vector, n_features)
        
        X_anormal, y_anormal, _, _, _, _ = utils.generate_sets(anormal_sequence, EncDec.parameters.get_n_steps(),input_form =EncDec.input_form, output_form = EncDec.output_form, validation=False, n_seq=EncDec.parameters.get_n_seq(), n_input=EncDec.parameters.get_n_input(), n_features=EncDec.parameters.get_n_features())
       
        anormal_dates = utils.generate_days(anormal_sequence, EncDec.parameters.get_n_input(), time, validation=False)
        print("test dates", len(anormal_dates))
        vector, y_pred = EncDec.predict_error(scaler_normal, model, X_anormal, y_anormal, y_anormal_inv, anormal_dates, "Time series test sequence")
        
        y_anormal_D = pd.DataFrame()
        y_anormal_D['value'] = y_anormal.ravel()
        y_anormal_D.index = anormal_dates
        
        events = network.getEvents()
        print(">>>>>>number of events", len(events))
        EncDec.file.append(Text.Text("Number of events year:" + str(len(events))))
            
        
        print("maxmin")
        start = min(y_anormal_D.index)
        end = max(y_anormal_D.index)
           
        if time=='date':
            start = change_format(start, '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S')
            end = change_format(end, '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S')
           
        events = network.select_events(start, end, time)
        
        print("number of events", len(events))
        EncDec.file.append(Text.Text("Number of events during %s to %s: %s" % 
                                     (start, end, len(events))))
        #min_th = test_betas(X_anormal, y_anormal_D, EncDec.h5_file_name, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), events, EncDec.file, time=time)
        vector, _ = EncDec.predict_error(scaler_normal, model, X_anormal, y_anormal, y_anormal_inv, anormal_dates, "Time series Normal and Anormal Sequence")        
        score = utils.anomaly_score(mu, sigma, vector, n_features) 
        
        min_th = get_threshold_f_beta_score(score, X_anormal, y_anormal_D, EncDec.h5_file_name, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), events, EncDec.file, time=time, beta=0.01, type_sequence='anormal')
    
        print("fbetascore threshold", min_th)
        
        
        #min_th, encoded = get_threshold_tailed(EncDec.file, score)
        #print("tailedscore threshold", min_th)
        #img = image.Image("Anomaly Score", encoded)
        #EncDec.file.append(img)
        EncDec.min_th = min_th
        EncDec.file.append(Text.Text("Threshold used:" + str(min_th)))
     
         
        if EncDec.parameters.get_save() == True:
           print("Save parameters")
           utils.save_parameters(scaler_normal, mu, sigma, EncDec.parameters.get_n_steps(), min_th, EncDec.h5_file_name)
       
       
    
        
        #plot all time series
        EncDec.plot_all_time_series()
        EncDec.all_timeseries = []
        
        
        return loss, val_loss
    
    def train(anormal_sequence_inv, normal_sequence_inv, network, k, config, file, train_chunks, anormal_chunks,time, type_model_func, toIndex, validation):
         h5_file_name = EncDec.h5_file_name
         n_features = EncDec.parameters.get_n_features()
         print("n_feature", n_features)
         n_input = EncDec.parameters.get_n_input()
         n_steps = EncDec.parameters.get_n_steps()
         n_seq = EncDec.parameters.get_n_seq()
         input_form = EncDec.input_form
         output_form = EncDec.output_form

         
         run_losses = list()
         run_val_losses = list()
         best_val_loss = None
         for i in range(k):  
                scaler = None
                scaler = utils.fit_data(train_chunks[i], scaler)
                train_chunks[i] = utils.transform_data(train_chunks[i], scaler)
                anormal_chunks[i] = utils.transform_data(anormal_chunks[i], scaler)
                EncDec.add_stat('Number of partition being trained:', i)
                print("PARTITION NO:>>>>>>>>", i)
                normal_sequence = train_chunks[i]
                anormal_sequence = anormal_chunks[i]
                sequence_inv = select_data(normal_sequence_inv, time, min(normal_sequence.index), max(normal_sequence.index))
                test_inv = select_data(anormal_sequence_inv, time, min(normal_sequence.index), max(normal_sequence.index))
                EncDec.verifySizes(normal_sequence, n_input, n_steps, n_seq, "train")
                
                size_train = len(normal_sequence)
                print("Training size:", size_train)
                EncDec.file.append(Text.Text("Training size:" + str(size_train)))
                start_train = min(normal_sequence.index)
                end_train = max(normal_sequence.index)
                EncDec.file.append(Text.Text("Date train events %s to %s" % 
                                              (start_train, end_train)))
            
 
                X_train_full, y_train_full = utils.generate_full(normal_sequence, n_steps, input_form=input_form, output_form=output_form, n_seq=n_seq,n_input=n_input, n_features=n_features)
                print("X_train_full shape", X_train_full.shape)
                print("y_train_full shape", y_train_full.shape)
                print("config", config)
                print("kernel size", config[2])
                
                EncDec.define_seeds()
                model = type_model_func(X_train_full, y_train_full, config)
                batch_size = tuning.get_param(config, toIndex, "batch_size")
          
                history = list()
           
                
                X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2  = utils.generate_sets(normal_sequence, n_steps,input_form = input_form, output_form = output_form, validation=validation, n_seq=n_seq,n_input=n_input, n_features=n_features)  
        
                dates_train, dates_val_1, dates_val_2 = utils.generate_days(normal_sequence, n_input, time)
                
                filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
                es = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=5, verbose=1)
                #mc = ModelCheckpoint(filepath, monitor='val_loss', mode='min', save_best_only=True)
                
                
                if validation:
                    if EncDec.split:
                        X_train = utils.split_features(EncDec.parameters.get_n_features(), X_train)
                        X_val_1 = utils.split_features(EncDec.parameters.get_n_features(), X_val_1)
                        
                    history = model.fit(X_train, y_train, validation_data=(X_val_1, y_val_1), epochs=200, batch_size=batch_size, callbacks=[es]).history
                else:
                    history = model.fit(X_train, y_train, epochs=200, batch_size=batch_size, callbacks=[es]).history
               
                plot_model(model, to_file='model.png')
                filename = EncDec.h5_file_name +'.h5'
                print("filename:", filename)
                path = os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", filename)
                model.save(path)
                print("Saved model to disk")
                
                model = load_model(path)
                _, _, _, y_val_1_inv,_, y_val_2_inv  = utils.generate_sets(sequence_inv, n_steps,input_form = input_form, output_form = output_form, validation=validation, n_seq=n_seq,n_input=n_input, n_features=n_features)  
                _, y_anormal_inv, _, _ ,_, _  = utils.generate_sets(sequence_inv, n_steps,input_form = input_form, output_form = output_form, validation=False, n_seq=n_seq,n_input=n_input, n_features=n_features)  
                try:
                    loss, val_loss = EncDec.test_stats(scaler, y_val_1_inv, y_val_2_inv, y_anormal_inv, X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2, X_train_full, y_train_full, model, network, anormal_sequence, normal_sequence, time, history)
                    run_losses.append(loss)
                    run_val_losses.append(val_loss)
                except ValueError:
                    pass
                if best_val_loss == None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    EncDec.best_h5_filename = "best_" + h5_file_name 
                    path = os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", EncDec.best_h5_filename + ".h5")
                    model.save(path)
                    if EncDec.parameters.get_save() == True:
                        print("Save best parameters")
                        utils.save_parameters(scaler, EncDec.mu, EncDec.sigma, EncDec.parameters.get_n_steps(), EncDec.min_th, EncDec.best_h5_filename)
       
                    
                K.clear_session()
                
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
            print("simulated", simulated)
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
    
    def get_number_of_cv_partitions(train_chunks):
        return len(train_chunks)
    
 
    def get_columns_to_exclude(simulated):
        to_exclude = []
        if simulated:
            to_exclude = ['leak']
        else:
            to_exclude = []
        return to_exclude
        
    def define_seeds():
        #python
        os.environ['PYTHONHASHSEED'] = '0'
        
        #seed for numpy generator seeds
        np.random.seed(17)
        
        #python generator numbers
        rn.seed(1254)
        
        #tensorflow random numbers
        tf.set_random_seed(37)
        
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        
        #for tensorflow to use a single thread
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        
        
        
        
    @classmethod
    def do_train(cls, user_parameters):
            EncDec.parameters = user_parameters
            simulated = EncDec.parameters.get_simulated()
            sensors=["12"]
            iterative=False
            cv=True
            typeData = EncDec.get_type_data(simulated)
            print("type_data", typeData)
            network = Network("infraquinta", typeData=typeData, chosen_sensors=sensors,
                              no_leaks=1000, load=True)
          
            time = EncDec.get_time_column(simulated)
            import os

            path_report = os.path.join(os.getcwd(), "wisdom/reports_files/report_models/","%s.html" % cls.report_name)

            EncDec.init_report(path_report)
           
            n_train = EncDec.parameters.get_n_train()
            EncDec.h5_file_name = cls.h5_file_name
            toIndex = cls.toIndex
            EncDec.file = cls.file
        
            type_model_func = cls.type_model_func
            
            config = cls.config
            to_exclude = EncDec.get_columns_to_exclude(simulated)
            map_stat = EncDec.do_map_stat(EncDec.parameters.get_hidden_size(), EncDec.parameters.get_code_size(), 
                                   EncDec.parameters.get_dropout(), EncDec.parameters.get_regularizer(), EncDec.parameters.get_n_steps())
            
            EncDec.input_form=cls.input_form
            EncDec.output_form=cls.output_form

            file = EncDec.add_multiple_stats(map_stat)
            
            stime = "2017-01-01 00:00:00"
            etime = "2018-09-30 23:59:00"
            
            all_sequence, all_normal_sequence = EncDec.generate_sequences_sensors(sensors, time, simulated, network, stime=stime, etime=etime)
            all_sequence = all_sequence.drop(to_exclude, axis=1)
            all_normal_sequence = all_normal_sequence.drop(to_exclude, axis=1)
            
            
            #transform data with minimum loose of information
            anormal_sequence_inv = all_sequence 
            all_sequence = utils.transform_sequence(all_sequence, adjustment=False)
            normal_sequence_inv = all_normal_sequence
            all_normal_sequence = utils.transform_sequence(all_normal_sequence, adjustment=False)
         
            EncDec.normal_sequence = all_normal_sequence
            print("all_sequence", all_sequence.columns)
            print("all_sequence", all_sequence.index)
            
            print("normal_sequence", all_normal_sequence.columns)
            print("normal_sequence", all_normal_sequence.index)
            
            
            
            if EncDec.parameters.get_bayesian() == True:
                cls.config = tuning.do_bayesian_optimization(cls.fitness, cls.dimensions, cls.default_parameters, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_seq())
            
            config = cls.config
            K.clear_session()   
        
            train_chunks, anormal_chunks = EncDec.split_train_test_dataframe(sensors, cv, n_train, all_normal_sequence, all_sequence, time, network.getEvents())
           
            k = EncDec.get_number_of_cv_partitions(train_chunks)
            
            EncDec.add_stat('Number of partitions:', k)
            print("NUMBER OF PARTITIONS:", k)
    
            EncDec.parameters.set_n_features(len(train_chunks[0].columns))
            cls.parameters = EncDec.parameters
            
            run_losses, run_val_losses = EncDec.train(anormal_sequence_inv, normal_sequence_inv, network, k, config, file, train_chunks, anormal_chunks,time, type_model_func, toIndex, EncDec.parameters.get_validation())
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
            prediction = utils.detect_anomalies(data,  EncDec.best_h5_filename)
            return prediction
        
    @classmethod
    def do_test(cls, user_parameters):
        EncDec.parameters = user_parameters
        simulated = EncDec.parameters.get_simulated()
        EncDec.input_form=cls.input_form
        EncDec.output_form=cls.output_form
        EncDec.best_h5_filename = "best_" + cls.h5_file_name 
        param = utils.load_parameters(EncDec.best_h5_filename)
        scaler = param['scaler']
        sensors=["12"]
        typeData = EncDec.get_type_data(simulated)
        print("type_data", typeData)
        network = Network("infraquinta", typeData=typeData, chosen_sensors=sensors,
                          no_leaks=1000, load=True)
        
        events = network.getEvents()
        
        for event in events:
            print("event", event.getStart())
        
        
        time = EncDec.get_time_column(simulated)  
        stime = "2018-01-30 00:00:00"
        etime = "2018-08-31 23:59:00"
        test_sequence, _ = EncDec.generate_sequences_sensors(sensors, time, simulated, network, stime=stime, etime=etime)
        
        _, y_inv, _, _, _, _ = utils.generate_sets(test_sequence, EncDec.parameters.get_n_steps(),input_form =EncDec.input_form, output_form = EncDec.output_form, validation=False, n_seq=EncDec.parameters.get_n_seq(), n_input=EncDec.parameters.get_n_input(), n_features=EncDec.parameters.get_n_features())
        test_sequence = utils.transform_data(test_sequence, scaler)
        X_test, y_test, _, _, _, _ = utils.generate_sets(test_sequence, EncDec.parameters.get_n_steps(),input_form =EncDec.input_form, output_form = EncDec.output_form, validation=False, n_seq=EncDec.parameters.get_n_seq(), n_input=EncDec.parameters.get_n_input(), n_features=EncDec.parameters.get_n_features())
       
        ###### DAYS
        test_dates = utils.generate_days(test_sequence, EncDec.parameters.get_n_input(), time, validation=False)
        y_test_D = pd.DataFrame()
        y_test_D['value'] = y_test.ravel()
        y_test_D.index = test_dates
        print("detect_anomalies")
        predictions = utils.detect_anomalies(X_test, y_test_D, y_inv, EncDec.best_h5_filename, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), choose_th=None, time='date')
        print("make_metrics")
   
        make_metrics(predictions, network.getEvents(), y_test_D, "date")
        print("blocks")
        test_blocks(test_sequence, network.getEvents(), y_inv, EncDec.best_h5_filename, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), metric='accuracy', time='date', choose_th=None, input_form=EncDec.input_form, output_form=EncDec.output_form, n_seq=EncDec.parameters.get_n_seq(), n_input=EncDec.parameters.get_n_input(), size_block=1200)
        
        return True