# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:45:57 2020

@author: anama
"""

from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from keras.models import Sequential
from keras.models import Model
from preprocessing.series import create_dataset_as_supervised, create_dataset, generate_sequences, series_to_supervised, generate_normal, change_format
from preprocessing.series import downsample, generate_total_sequence, select_data, csv_to_df
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
from keras.utils.vis_utils import plot_model
from kstest import goodness_of_fit, ecdf
from architecture.timeseries import timeseries
from architecture.parameters import parameters
from preprocessing.seasonality import inverse_difference
from architecture.corrective_model import temperature_correction_factor, df_info_temperature
from evaluation.metrics import confusion_matrix, getTP, getFP, getFN, getTN, make_metrics
from evaluation.bootstraping import test_blocks
from evaluation.graphs import bars_leaks
import random as rn
import tensorflow as tf
from keras import backend as K
from architecture.threshold import get_threshold_tailed, algorithm_2T, IQR, median_absolute_deviation, standard_deviation, search_fbs, search_theshold_tailed, probability_event, chebyshev_contract
import architecture.reconstruction_error
from report.Report import report
from architecture.loss_function import chebyshev, euclidean_distance
import keras.losses
#keras.losses.custom_loss = chebyshev
from utils import save_model_json, load_model_json
from tuning import save_config, load_config
from preprocessing.series import subtrair_consumos
class EncDec(object):
    config = []
    mu = 0
    sigma = 0
    th_min = 0
    th_max = 0
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
    temperature_factor = None
    events = None
    network = None
    sensors = None
    n_train = 0
    n_test = 0
    iterative=False
    cv=True
    path_report = None
    type_model_func = None
    config = None
    to_exclude = None
    map_stat = None
    model_name = None
    

    def setNetwork():
        simulated = EncDec.parameters.get_simulated()
        typeData = EncDec.get_type_data(simulated)
        EncDec.network = Network("infraquinta", typeData=typeData, chosen_sensors=EncDec.sensors,
                              no_leaks=EncDec.parameters.get_n_leaks(), load=True)
        #events = Network.filterEventsByCloseSensor(EncDec.network.getEvents(), ['10'])
        #print(len(events))
        #events = Network.filterEventsByCloseSensor(EncDec.network.getEvents(), ['2'])
        #print(len(events))
        
    
    def setEvents(start, end, time, filter_by = None):
        events = EncDec.network.getEvents()
        #events = Network.filterByDate(events, start, end, time)
        
        if filter_by == None:
            EncDec.events = events
        elif filter_by == "Close Sensor":
            EncDec.events = Network.filterEventsByCloseSensor(events, EncDec.sensors)
        elif filter_by == "Avg Flow": 
            EncDec.events = Network.filterByAvgFlow(events, size="small")
    
    def set_layers_nodes(input_size, type_autoencoder="overcomplete", constraints=["fixed decoder"], config_node="medium", no_fixed_nodes=12):
            #number of nodes for each layer
            nodes_layers = []
            #bottleneck layer
            code_size = 0
            #small, medium, big number nodes to decrease each layer
            no_nodes_configs = {'small':[4,16], 'medium':[32,64], 'big':[64,128], 'very big':[128, 256]}
            nodeconfig = no_nodes_configs[config_node]
            code_size = nodeconfig[0]
            hidden_size = nodeconfig[1]
            if type_autoencoder == "overcomplete":
                if code_size > input_size:
                    print("overcomplete")
                else:
                    raise ValueError("Not overcomplete")
            elif type_autoencoder == "undercomplete":
                if code_size < input_size:
                    print("undercomplete")
                else:
                    raise ValueError("Not undercomplete")
            elif type_autoencoder == "fixed":
                print("fixed")
                no_nodes = no_fixed_nodes
           
                    
            if "fixed decoder" in constraints:
                no_nodes = no_fixed_nodes
               

            return hidden_size, code_size
        
    @classmethod
    def setup(cls, user_parameters, type_setup="train"):
         EncDec.model_name = cls.model_name
         EncDec.parameters = user_parameters
         simulated = EncDec.parameters.get_simulated()
         EncDec.sensors=EncDec.parameters.get_sensors()
         EncDec.setNetwork()
       
         import os
         EncDec.path_report = os.path.join(os.getcwd(), "wisdom/reports_files/report_models/","%s.html" % cls.report_name)
         EncDec.rep = report()
         EncDec.n_train = EncDec.parameters.get_size_train()
         EncDec.n_test = EncDec.parameters.get_size_test()
         EncDec.h5_file_name = cls.h5_file_name
         EncDec.toIndex = cls.toIndex
         EncDec.file = HtmlFile.HtmlFile()
         EncDec.file = cls.file
         EncDec.type_model_func = cls.type_model_func
         EncDec.config = cls.config
         EncDec.to_exclude = EncDec.get_columns_to_exclude(simulated)
         map_stat = EncDec.do_map_stat(EncDec.parameters.get_hidden_size(), EncDec.parameters.get_code_size(), 
                                   EncDec.parameters.get_dropout(), EncDec.parameters.get_regularizer(), EncDec.parameters.get_n_steps())
            
         EncDec.input_form=cls.input_form
         EncDec.output_form=cls.output_form
         EncDec.add_multiple_stats(map_stat)
         EncDec.best_h5_filename = "best_" + cls.h5_file_name 
         EncDec.time = EncDec.get_time_column(simulated)  
         
         if type_setup == "train":
             EncDec.stime = "2017-01-01 00:00:00"
             EncDec.etime = "2018-09-30 23:59:00"
             EncDec.parameters.set_granularity('15min')
             if simulated:
                    EncDec.stime = pd.Timedelta(0, unit='days')
                    EncDec.etime = pd.Timedelta(730, unit='days')
                  
                    EncDec.is_temperature_correction = False
                    EncDec.parameters.set_granularity('10min')
         elif type_setup == "test":
             EncDec.path_report = os.path.join(os.getcwd(), "wisdom/reports_files/report_models/","%s.html" % (cls.report_name + "_test"))
             EncDec.rep = report()
             EncDec.best_h5_filename = "best_" + cls.h5_file_name 
             EncDec.param = utils.load_parameters(EncDec.best_h5_filename)
             EncDec.scaler = EncDec.param['scaler']
             
             EncDec.time = EncDec.get_time_column(simulated)  
             EncDec.stime = "2018-01-30 00:00:00"
             EncDec.etime = "2018-08-31 23:59:00"
            
        
             if simulated:
                 EncDec.stime = pd.Timedelta(730, unit='days')
                 EncDec.etime = pd.Timedelta(876, unit='days')
                 EncDec.parameters.set_granularity('10min')
                 EncDec.time = 'time'
         
         time = EncDec.get_time_column(simulated)
         EncDec.setEvents(EncDec.stime, EncDec.etime, time, filter_by = "Close Sensor")
         EncDec.set_layers_nodes(EncDec.parameters.get_n_input(), type_autoencoder="fixed", constraints=["fixed decoder"], config_node="medium", no_fixed_nodes=12)
        
 
    
    def add_timeseries(timeseries):
        EncDec.all_timeseries.append(timeseries)
        
    def generate_sequences_sensors(sensors, time, simulated, network, stime=None, etime=None, normal_simulation_id=2):
        """
        Parameters
        ----------
        sensors : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        print("generate_sequences_sensors")
        print("stime", stime)
        print("etime", etime)
        all_normal_sequence = list()
        all_sequence = list()
        print("NUMBER LEAKS NETWORKS", network.getNoLeaks())
        print("sensors", sensors)
        for sensor_id in sensors:
                print("sensor_id", sensor_id)
                normal_sequence, _ = generate_sequences(sensor_id, "sensortgmeasurepp",column_time=time, simulated=simulated, df_to_csv=True, no_leaks=network.getNoLeaks(), start=stime, end=etime, normal_simulation_id=normal_simulation_id)
                all_normal_sequence.append(normal_sequence)
                sequence = generate_total_sequence(sensor_id, "sensortgmeasurepp",start=stime, end=etime, simulated=simulated, no_leaks=network.getNoLeaks())
                all_sequence.append(sequence)

        #column-wise   
        df_union_sequence = pd.concat(all_sequence, axis=1)   
        df_union_normal = pd.concat(all_normal_sequence, axis=1)  
        
        print("df union index", df_union_sequence.index)
        return df_union_sequence, df_union_normal
    
    def split_train_test_dataframe(sensors, cv, n_train, n_test, all_normal_sequence, all_sequence, time):
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
        print("len", len(all_normal_sequence))
        print("events", len(EncDec.events))
     
        if cv:
                print("TIME", time)
                train_chunks, test_chunks = expanding_window(all_normal_sequence, all_sequence, n_train, n_test, EncDec.events, time=time, test_split=0.33)
        
        print("train_chunks", len(train_chunks))
        print("test_chunks", len(test_chunks))
        print("train", len(train_chunks[0]))
        print("test", len(test_chunks[0]))
        print("train", len(train_chunks[1]))
        print("test", len(test_chunks[1]))
   
       
    
        
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
    

    
  
    def plot_all_time_series():
        for ts in EncDec.all_timeseries:
            print(ts.get_dates())
            utils.plot_series(ts.get_name(), ts.get_dates(), ts.get_y_true(), ts.get_y_pred(), EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), EncDec.rep,  ts.get_y_inv(), EncDec.scaler, adjustment=EncDec.parameters.get_seasonality())
            
            
    def graph_components(map_id_avg, threshold):
        from evaluation import graphs
        graphs.bars_leaks(map_id_avg, threshold)
       
   
    def training_losses(model, X, y, dates,  title):
        y_pred = model.predict(X)
        y_pred = utils.process_predict(y_pred,  EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features())
        y_pred = pd.DataFrame(y_pred)
        scored = pd.DataFrame(index=y_pred.index)
        ytrain = utils.process_predict(y,  EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features())
        struct = {'y_pred': y_pred, 'ytrain': ytrain}
        dbfile = open('struct', 'ab')   
        # source, destination 
        pickle.dump(struct, dbfile)                      
        dbfile.close() 
        scored['Loss_mae'] = np.mean(np.abs(y_pred-ytrain), axis = 1)
        scored['Loss_ae'] = np.abs(y_pred-ytrain)
        print("loss ae shape", scored['Loss_ae'].shape)
        print("loss mae shape", scored['Loss_mae'].shape)
        utils.plot_bins_loss('Loss Distribution', scored['Loss_ae'], EncDec.rep)
      
        goodness_of_fit(EncDec.rep, "Loss Absolute error", scored['Loss_ae'], alpha=0.05)
        ecdf(scored['Loss_ae'], EncDec.rep)
    
        architecture.reconstruction_error.plots(title, scored['Loss_ae'], dates)
        
     
     
        
    
    
    
   
    
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
     
  
        
    def make_prediction(model, X, y, temperature_correction=False):
        #have days
        y_pred = model.predict(X)
        
        df = pd.DataFrame()
        df['value_true'] = y['value']
        df['value_pred'] = y_pred.ravel()
        days = 5
        #construct dataframe with temperatures, values, value average past days and change temperatures columns
        if temperature_correction:  
         
            print("len y_pred", len(y_pred))
            print("len y_true", len(y))
            
            df = df_info_temperature(y, y_pred, days)
            for index, row in df.iterrows():
                #desvio de temperatura a considerar
                dev = 3
                avg_past_days = 'Value_Average_Past_%s_days' % (days)
                if row['value_pred'] > row[avg_past_days] + dev or row['value_pred'] < row[avg_past_days] - dev:
                    #reject forecast and correct it
                    df.at[index, 'value_pred'] = row['value_pred'] * EncDec.temperature_correction * row['diff']
            
        return  df
    
    def predict_error(scaler_normal, model, X_val, y_val, y_inv, dates, title, mode="normal", metric="ae"):
        print("y_true", type(y_val))
        
        df = EncDec.make_prediction(model, X_val, y_val, temperature_correction=EncDec.is_temperature_correction)
        #y_pred = pd.DataFrame()
        #y_val = pd.DataFrame()
        y_pred = df['value_pred'].values.reshape(-1, 1)
        #y_pred.index = df.index
        y_val = df['value_true'].values.reshape(-1, 1)
        #y_val.index = df.index
        
        """
        if isinstance(df.index, pd.TimedeltaIndex):
            dates = df.index.to_pytimedelta()
        else:
            dates = df.index.values
        """
       
        for n in range(EncDec.parameters.get_n_features()):
            dates = df.index
            ts = timeseries(title, dates, y_val[:, n], y_pred[:, n], y_inv[:,n])
            EncDec.add_timeseries(ts) 
            
       
        #y_pred = model.predict(X_val)
   
        vector = utils.get_error_vector(y_val, y_pred, y_inv, scaler_normal, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), adjustment=EncDec.parameters.get_seasonality(), metric=metric)
      
      
        y_val = df[['value_true']]
        y_val = y_val.rename(columns={'value_true':'value'})
        print("y_val columns", y_val.columns)
    
        return vector, y_pred, y_val
        
   
    def test_stats(scaler_normal, y_val_1_inv, y_val_2_inv, y_anormal_inv, X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2, X_train_full, y_train_full, model, network, anormal_sequence,  normal_sequence, time, history, split):
        loss, val_loss = EncDec.get_losses(history)
        print("SPLIT", split)
        dates_train, dates_val_1, dates_val_2 = utils.generate_days(normal_sequence, EncDec.parameters.get_n_input(), time, n_val_sets=EncDec.parameters.get_n_val_sets(), train_split=split)
        print("split", split)
        print("dates train", dates_train)
        print("dates_val_1", len(dates_val_1))
        print("y_val_1", len(y_val_1))
     
        n_features = EncDec.parameters.get_n_features()
        EncDec.rep.add_text("Loss last epoch:" + str(loss))
        EncDec.rep.add_text("Validation loss last epoch:" + str(val_loss))
                 
        if EncDec.validation == True:
            utils.plot_training_losses(EncDec.rep, history)    
        dates_full = utils.generate_days(normal_sequence, EncDec.parameters.get_n_input(), time, validation=False, train_split=split)
        EncDec.training_losses(model, X_train_full, y_train_full, dates_full,  "Full sequence")
        y_val_1_D = pd.DataFrame()
        y_val_1_D['value'] = y_val_1.ravel()
        y_val_1_D.index = dates_val_1
        print(">>>>>>>>>>>>>>>>>>>>VALIDATION SET 1")
        #calculate loss on the validation set 1 to get miu and sigma values on validation set 1
        vector, y_pred, y_val_1_D = EncDec.predict_error(scaler_normal, model, X_val_1, y_val_1_D, y_val_1_inv, dates_val_1, "Time series validation set 1")
   
        EncDec.training_losses(model, X_val_1, y_val_1, dates_val_1, "Training losses graph for validation set 1")
        
        mu = utils.get_mu(vector)
        EncDec.mu = mu
        sigma = utils.get_sigma(vector, mu) 
        EncDec.sigma = sigma
        
        timeseries.do_histogram(vector)
        score = utils.anomaly_score(mu, sigma, vector, n_features, type_score = EncDec.parameters.getTypeScore())
        
        if not(X_val_2.size == 0):
            y_val_2_D = pd.DataFrame()
            y_val_2_D['value'] = y_val_2.ravel()
            y_val_2_D.index = dates_val_2
            print(">>>>>>>>>>>>>>>>>>>>VALIDATION SET 2")
            #calculate new error vector for validation set 2 
            vector, y_pred, y_val_2_D = EncDec.predict_error(scaler_normal, model, X_val_2, y_val_2_D, y_val_2_inv, dates_val_2, "Time series validation set 2")        
            score = utils.anomaly_score(mu, sigma, vector, n_features)
            
      
        #### SAVE TEMPORARILY MODEL
        if EncDec.parameters.get_save() == True:
           print("Save parameters")
           utils.save_parameters(scaler_normal, mu, sigma, EncDec.parameters.get_n_steps(), None, None, EncDec.h5_file_name, None, None)
       
       
        
        print(">>>>>>>>>>>>>>>>>>>>ANORMAL SET")
        X_anormal, y_anormal, _, _, _, _ = utils.generate_sets(anormal_sequence, EncDec.parameters.get_n_steps(),input_form =EncDec.input_form, output_form = EncDec.output_form, validation=False, n_seq=EncDec.parameters.get_n_seq(), n_input=EncDec.parameters.get_n_input(), n_features=EncDec.parameters.get_n_features(), n_val_sets = EncDec.parameters.get_n_val_sets(), train_split=split)
        anormal_dates = utils.generate_days(anormal_sequence, EncDec.parameters.get_n_input(), time, validation=False, train_split=split)
        y_anormal_D = pd.DataFrame()
        y_anormal_D['value'] = y_anormal.ravel()
        y_anormal_D.index = anormal_dates
        print("len anormal set", len(y_anormal_D))
        print("len train", len(dates_train))
     
        vector, y_pred, y_anormal_D = EncDec.predict_error(scaler_normal, model, X_anormal, y_anormal_D, y_anormal_inv, anormal_dates, "Time series Normal and Anormal Sequence")        
        score = utils.anomaly_score(mu, sigma, vector, n_features) 
        EncDec.rep.add_text("Number of events total events:" + str(len(EncDec.events)))
            
        start = min(y_anormal_D.index)
        end = max(y_anormal_D.index)
           
        if time=='date':
            start = change_format(start, '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S')
            end = change_format(end, '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S')
           
        EncDec.rep.add_text("Number of events during %s to %s: %s" % 
                                     (EncDec.stime, EncDec.etime, len(EncDec.events)))
  
        per = 1 - probability_event(EncDec.events, y_anormal_D.index)
        th_max, EncDec.fpr, EncDec.tpr = search_fbs(EncDec.rep, 0.1, 20, 0.1, score, X_anormal, y_anormal_D,  EncDec.h5_file_name, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), EncDec.events, time=EncDec.time, beta = 0.01, type_sequence='anormal')
        th_min = None
        chebyshev_contract(EncDec.rep, score)
        #th_min, th_max = search_theshold_tailed(EncDec.rep, score, X_anormal, y_anormal_D,  EncDec.h5_file_name, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), EncDec.events, EncDec.rep, time=time, beta = 0.01, type_sequence='anormal', start_percentage=0.001, end_percentage=0.1, margin=0.001, type_score=EncDec.parameters.getTypeScore())
        print(">>>>>>th_min", th_min)
        print(">>>>>>th_max", th_max)
        score, predictions = utils.detect_anomalies(X_anormal, y_anormal_D, y_anormal_inv, EncDec.h5_file_name, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), th_min, th_max, time=time, metric="ae")
        print("make_metrics")
        make_metrics(EncDec.rep, predictions, EncDec.events, y_anormal_D, time)
        EncDec.th_min = th_min
        EncDec.th_max = th_max
        EncDec.rep.add_text("Min Threshold used:" + str(th_min))
        EncDec.rep.add_text("Max Threshold used:" + str(th_max))
       
         
        if EncDec.parameters.get_save() == True:
           print("Save parameters")
           utils.save_parameters(scaler_normal, mu, sigma, EncDec.parameters.get_n_steps(), th_min, th_max, EncDec.h5_file_name, EncDec.fpr, EncDec.tpr)
        
        #plot all time series
        EncDec.plot_all_time_series()
        EncDec.all_timeseries = []
        
       
    
        
     
    
    def train(anormal_sequence_inv, normal_sequence_inv, network, k, config, train_chunks, anormal_chunks,time, type_model_func, toIndex, validation):
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
                  
         k = EncDec.get_number_of_cv_partitions(train_chunks)
         do_train = True
         if do_train:
             for i in range(k):  
                    scaler = None
                    size_anormal = len(anormal_chunks[i])
                    size_normal = len(train_chunks[i])
                    split = (size_normal)/(size_normal+size_anormal)
                    print("split", split)
                  
                    EncDec.scaler = utils.fit_data(train_chunks[i], scaler)
                    train_chunks[i] = utils.transform_data(train_chunks[i], EncDec.scaler)
                    anormal_chunks[i] = utils.transform_data(anormal_chunks[i], EncDec.scaler)
                   
                    scaler = EncDec.scaler
                    EncDec.add_stat('Number of partition being trained:', i)
                    print("PARTITION NO:>>>>>>>>", i)
                    normal_sequence = train_chunks[i]
                    anormal_sequence = anormal_chunks[i]
                    EncDec.rep.add_text("Size train chunk" + str(len(normal_sequence)))
                    EncDec.rep.add_text("Size anormal chunk" + str(len(normal_sequence)))
                    
                    fig1, axs = plt.subplots(2)
                    axs[0].plot(normal_sequence)
                    axs[1].plot(anormal_sequence)
                    fig1.savefig("timeseries.png")
                    
                    sequence_inv = select_data(normal_sequence_inv, time, min(normal_sequence.index), max(normal_sequence.index))
                    test_inv = select_data(anormal_sequence_inv, time, min(normal_sequence.index), max(normal_sequence.index))
                    EncDec.verifySizes(normal_sequence, n_input, n_steps, n_seq, "train")
                    
                    size_train = len(normal_sequence)
                    print("Training size:", size_train)
                    EncDec.rep.add_text("Training size:" + str(size_train))
                    start_train = min(normal_sequence.index)
                    end_train = max(normal_sequence.index)
                    print("Start train:", start_train)
                    print("End train:", end_train )
                    EncDec.rep.add_text("Date train events %s to %s" % 
                                                  (start_train, end_train))
                
                    print("n_input", n_input)
                    X_train_full, y_train_full = utils.generate_full(normal_sequence, n_steps, input_form=input_form, output_form=output_form, n_seq=n_seq,n_input=n_input, n_features=n_features)
                    
                    EncDec.define_seeds()
                    model = type_model_func(X_train_full, y_train_full, config)
                    batch_size = tuning.get_param(config, toIndex, "batch_size")
              
                    history = list()
               
                    _, y_train_inv, _, _, _, _  = utils.generate_sets(normal_sequence_inv, n_steps,input_form = input_form, output_form = output_form, validation=validation, n_seq=n_seq,n_input=n_input, n_features=n_features, n_val_sets=EncDec.parameters.get_n_val_sets(), train_split=split)       
                    X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2  = utils.generate_sets(normal_sequence, n_steps,input_form = input_form, output_form = output_form, validation=validation, n_seq=n_seq,n_input=n_input, n_features=n_features, n_val_sets = EncDec.parameters.get_n_val_sets(), train_split=split)  
            
                    dates_train, dates_val_1, dates_val_2 = utils.generate_days(normal_sequence, n_input, time,  n_val_sets=EncDec.parameters.get_n_val_sets(), train_split = split)
                    
                  
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
                    
                    y_pred = model.predict(X_train)
                    #vector_train = utils.get_error_vector(y_train, y_pred, y_train_inv, scaler, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), adjustment=EncDec.parameters.get_seasonality(), metric="ae")
          
             
                    #vector_train, _ = EncDec.predict_error(scaler, model, X_train, y_train, y_train_inv, dates_train, "Time series Train set")
                    #EncDec.temperature_factor = temperature_correction_factor(vector_train, dates_train, 'date')
                    #print("Temperature factor", EncDec.temperature_factor)
    
                    plot_model(model, to_file=os.path.join(os.getcwd(), "wisdom/architecture/models", EncDec.model_name + ".png" ), show_shapes=True, show_layer_names=False)
                    # Save model
                    save_model_json(model, EncDec.h5_file_name )
                    print("Saved model to disk")
                 
                    model = load_model_json(EncDec.h5_file_name)
                    
                    loss, val_loss = 5000, 5000
                    try:
                        loss, val_loss = EncDec.get_losses(history)
                        run_losses.append(loss)
                        run_val_losses.append(val_loss)
                    except ValueError:
                        pass
                    if best_val_loss == None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        EncDec.scaler = scaler
                        EncDec.best_h5_filename = "best_" + h5_file_name 
                        path = os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", EncDec.best_h5_filename + ".h5")
                        #model.save(path)
                        save_model_json(model, EncDec.best_h5_filename)
                        utils.save_train_parameters(EncDec.best_h5_filename, normal_sequence, anormal_sequence, sequence_inv, scaler, history, split)
           
                      
                        
                    K.clear_session()
             utils.save_cv_parameters(EncDec.best_h5_filename, run_losses, run_val_losses)
              
         #find threshold only for best model
         model = load_model_json(EncDec.best_h5_filename)
         param = utils.load_train_parameters(EncDec.best_h5_filename)
         normal_sequence = param['normal sequence']
         anormal_sequence = param['anormal sequence']
         sequence_inv = param['sequence']
         EncDec.scaler = param['scaler']
         history = param['history']
         split = param['split']
         EncDec.rep.add_text("Train split:" + str(split))
         
         
         X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2  = utils.generate_sets(normal_sequence, n_steps,input_form = input_form, output_form = output_form, validation=validation, n_seq=n_seq,n_input=n_input, n_features=n_features, n_val_sets = EncDec.parameters.get_n_val_sets(), train_split=split)  
          
         _, _, _, y_val_1_inv,_, y_val_2_inv  = utils.generate_sets(sequence_inv, n_steps,input_form = input_form, output_form = output_form, validation=validation, n_seq=n_seq,n_input=n_input, n_features=n_features, n_val_sets = EncDec.parameters.get_n_val_sets(), train_split=split)  
         _, y_anormal_inv, _, _ ,_, _  = utils.generate_sets(sequence_inv, n_steps,input_form = input_form, output_form = output_form, validation=False, n_seq=n_seq,n_input=n_input, n_features=n_features, n_val_sets = EncDec.parameters.get_n_val_sets(), train_split=split)  
                
         X_train_full, y_train_full = utils.generate_full(normal_sequence, n_steps, input_form=input_form, output_form=output_form, n_seq=n_seq,n_input=n_input, n_features=n_features)
                
         EncDec.test_stats(EncDec.scaler, y_val_1_inv, y_val_2_inv, y_anormal_inv, X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2, X_train_full, y_train_full, model, network, anormal_sequence, normal_sequence, time, history, split)
         
         if EncDec.parameters.get_save() == True:
             print("Save best parameters")
             utils.save_parameters(EncDec.scaler, EncDec.mu, EncDec.sigma, EncDec.parameters.get_n_steps(), EncDec.th_min, EncDec.th_max, EncDec.best_h5_filename, EncDec.fpr, EncDec.tpr)
       
         param_cv = utils.load_cv_parameters(EncDec.best_h5_filename)
         run_losses = param_cv['run losses'] 
         run_val_losses = param_cv['run val losses']
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
         EncDec.rep.add_text(stat_name + ':' + str(stat))
         
         
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
        
    def subtrair_sensores(all_sequence_1, all_normal_sequence_1):
     
        all_sequence_2, all_normal_sequence_2 = EncDec.generate_sequences_sensors(['6'], EncDec.time, EncDec.parameters.get_simulated(), EncDec.network, stime=EncDec.stime, etime=EncDec.etime, normal_simulation_id= EncDec.parameters.get_normal_simulation_id())
        all_sequence_2 = all_sequence_2.drop(EncDec.to_exclude, axis=1)
        
        all_sequence_3, all_normal_sequence_3 = EncDec.generate_sequences_sensors(['12'], EncDec.time, EncDec.parameters.get_simulated(), EncDec.network, stime=EncDec.stime, etime=EncDec.etime, normal_simulation_id= EncDec.parameters.get_normal_simulation_id())
        all_sequence_3 = all_sequence_3.drop(EncDec.to_exclude, axis=1)
        
        new_sequence, new_normal_sequence = subtrair_consumos(all_sequence_1, all_sequence_2, all_normal_sequence_1, all_normal_sequence_2)
        print("all_sequence", new_sequence.shape)
        print("normal_sequence", new_normal_sequence.shape)
        all_sequence, all_normal_sequence = subtrair_consumos(new_sequence, all_sequence_3, new_normal_sequence, all_normal_sequence_3) 
        print("all_sequence", all_sequence.shape)
        print("normal_sequence", all_normal_sequence.shape)
        return all_sequence, all_normal_sequence
            
            
        
        
    @classmethod
    def do_train(cls, user_parameters):
            cls.setup(user_parameters)
            print("stime", EncDec.stime)
            print("etime", EncDec.etime)
            all_sequence, all_normal_sequence = EncDec.generate_sequences_sensors(EncDec.sensors, EncDec.time, EncDec.parameters.get_simulated(), EncDec.network, stime=EncDec.stime, etime=EncDec.etime, normal_simulation_id= EncDec.parameters.get_normal_simulation_id())
            all_sequence = all_sequence.drop(EncDec.to_exclude, axis=1)
           
            ###testar subtrair consumos
            #all_sequence, all_normal_sequence = EncDec.subtrair_sensores(all_sequence, all_normal_sequence)
           
            if 'leak' in all_normal_sequence.columns:
                all_normal_sequence = all_normal_sequence.drop(EncDec.to_exclude, axis=1)
            print("normal_sequence", len(all_normal_sequence))
            #transform data with minimum loose of information
            anormal_sequence_inv = all_sequence 
            all_sequence = utils.transform_sequence(all_sequence, adjustment=EncDec.parameters.get_seasonality(), granularity=EncDec.parameters.get_granularity(), lag=EncDec.parameters.get_lag_seasonality())
            normal_sequence_inv = all_normal_sequence
            all_normal_sequence = utils.transform_sequence(all_normal_sequence, adjustment=EncDec.parameters.get_seasonality(), granularity=EncDec.parameters.get_granularity(), lag=EncDec.parameters.get_lag_seasonality())
         
            EncDec.normal_sequence = all_normal_sequence
            print("normal_sequence", len(all_normal_sequence))
        
            if EncDec.parameters.get_bayesian() == True:
                cls.config = tuning.do_bayesian_optimization(cls.fitness, cls.dimensions, cls.config, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_seq(), EncDec.toIndex)
                save_config(cls.config, cls.model_name)
                print("config", cls.config)
                print("Bayesian configuration saved")
               
            
            if EncDec.parameters.get_load_bayesian() == True:
                cls.config = load_config(cls.model_name)
                
            EncDec.config = cls.config
            K.clear_session()   
            
            n_train = EncDec.parameters.get_size_train()
            n_test = EncDec.parameters.get_size_test()
            train_chunks, anormal_chunks = EncDec.split_train_test_dataframe(EncDec.sensors, EncDec.cv, n_train, n_test, all_normal_sequence, all_sequence, EncDec.time)
            #train_chunks = [train_chunks[0]]
            #anormal_chunks = [anormal_chunks[0]]
            k = EncDec.get_number_of_cv_partitions(train_chunks)
            
            EncDec.add_stat('Number of partitions:', k)
            print("NUMBER OF PARTITIONS:", k)
    
            EncDec.parameters.set_n_features(len(train_chunks[0].columns))
            cls.parameters = EncDec.parameters
            
            run_losses, run_val_losses = EncDec.train(anormal_sequence_inv, normal_sequence_inv, EncDec.network, k, EncDec.config, train_chunks, anormal_chunks,EncDec.time, EncDec.type_model_func, EncDec.toIndex, EncDec.parameters.get_validation())
            mean_loss, mean_val_loss = EncDec.do_means(run_losses, run_val_losses)
            
            map_stat = {'Mean loss': mean_loss, 'Mean validation loss': mean_val_loss}
            EncDec.add_multiple_stats(map_stat)
            EncDec.rep.write_report(EncDec.path_report)
            
            return True
    
    
    def operation(data, anomaly_threshold):
            prediction = utils.detect_anomalies(data,  EncDec.best_h5_filename)
            return prediction
        
    @classmethod
    def do_test(cls, user_parameters):
        print(">>>>>>DO TEST")
        cls.setup(user_parameters, type_setup = "test")
    
        test_sequence, _ = EncDec.generate_sequences_sensors(EncDec.sensors, EncDec.time, EncDec.parameters.get_simulated(), EncDec.network, stime=EncDec.stime, etime=EncDec.etime)
        #test_sequence, _ = EncDec.subtrair_sensores(test_sequence, pd.DataFrame())
        
        param = utils.load_train_parameters(EncDec.best_h5_filename)
        size_normal_sequence = param['normal sequence'].shape[0]
        #anormal_sequence = param['anormal sequence']
        size_block_test = int(round(0.2*size_normal_sequence))
        #test_sequence = test_sequence[:int(round(0.2*size_normal_sequence))] 
         
        #transform data with minimum loose of information
        test_sequence_inv = test_sequence 
        test_sequence = utils.transform_sequence(test_sequence, adjustment=EncDec.parameters.get_seasonality(), granularity=EncDec.parameters.get_granularity(), lag=EncDec.parameters.get_lag_seasonality())
         
        _, y_inv, _, _, _, _ = utils.generate_sets(test_sequence_inv, EncDec.parameters.get_n_steps(),input_form =EncDec.input_form, output_form = EncDec.output_form, validation=False, n_seq=EncDec.parameters.get_n_seq(), n_input=EncDec.parameters.get_n_input(), n_features=EncDec.parameters.get_n_features(), n_val_sets = EncDec.parameters.get_n_val_sets())
        test_sequence = utils.transform_data(test_sequence, EncDec.scaler)
        X_test, y_test, _, _, _, _ = utils.generate_sets(test_sequence, EncDec.parameters.get_n_steps(),input_form =EncDec.input_form, output_form = EncDec.output_form, validation=False, n_seq=EncDec.parameters.get_n_seq(), n_input=EncDec.parameters.get_n_input(), n_features=EncDec.parameters.get_n_features(), n_val_sets = EncDec.parameters.get_n_val_sets())
       
        ###### DAYS
        test_dates = utils.generate_days(test_sequence, EncDec.parameters.get_n_input(), EncDec.time, validation=False)
        y_test_D = pd.DataFrame()
        y_test_D['value'] = y_test.ravel()
        y_test_D.index = test_dates
        print("detect_anomalies")
        print("best", EncDec.best_h5_filename)
        score, predictions = utils.detect_anomalies(X_test, y_test_D, y_inv, EncDec.best_h5_filename, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), EncDec.param['th_min'], EncDec.param['th_max'], time=EncDec.time, metric="ae")
        dates_scores = test_dates
        print("make_metrics")
        bars_leaks(EncDec.rep, score, dates_scores, EncDec.events[:8], EncDec.param['th_min'], EncDec.param['th_max'])
        EncDec.rep.add_text("Min Threshold used:" + str(EncDec.param['th_min']))
        EncDec.rep.add_text("Max Threshold used:" + str(EncDec.param['th_max']))
       
        make_metrics(EncDec.rep, predictions, EncDec.events, y_test_D, EncDec.time)
        #size_block_test = int(round(test_sequence.shape[0] / 5))
        test_blocks(EncDec.rep, test_sequence, EncDec.events, y_inv, EncDec.best_h5_filename, EncDec.parameters.get_n_steps(), EncDec.parameters.get_n_features(), metric='accuracy', time=EncDec.time, th_min=EncDec.param['th_min'], th_max=EncDec.param['th_max'], input_form=EncDec.input_form, output_form=EncDec.output_form, n_seq=EncDec.parameters.get_n_seq(), n_input=EncDec.parameters.get_n_input(), size_block=size_block_test)
        EncDec.rep.write_report(EncDec.path_report)
        return True