# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:42:07 2020

@author: anama
"""
from preprocessing.series import generate_sequences
import utils
import tuning
from keras.backend import clear_session
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
class Baseline(object):
    type_model_func = None
    type_model = ""
    n_steps = 3
    n_features = 1
    n_input = 1
    toIndex = dict()
    
    @classmethod
    def do_train(cls, timesteps=3, simulated = False, bayesian=False, save=True, validation=True):
            print("do_train")
            print("validation", validation)
            print("function", cls.type_model_func)
            print("to index", cls.toIndex)
          
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
            print("config", cls.config)
            model = cls.type_model_func(X_train_full, y_train_full, config)
            best_h5_filename = "best_" + cls.h5_file_name + ".h5"
            for df_chunk in normal_sequence:
                if is_best_model:
                    model = load_model(best_h5_filename)
                number_of_chunks += 1
                print("number of chunks:", number_of_chunks)
                X_train, y_train, X_val_1, y_val_1, X_val_2, y_val_2 = utils.generate_sets(df_chunk, timesteps,input_form = cls.input_form, output_form = cls.output_form, validation=validation, n_seq=cls.n_seq,n_input=cls.n_input, n_features=cls.n_features)  
                print("X_train shape", X_train.shape)
                print("y_train shape", y_train.shape)
                es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
                mc = ModelCheckpoint(best_h5_filename, monitor='val_loss', mode='min', save_best_only=True)
                if validation:
                    history = model.fit(X_train, y_train, validation_data=(X_val_1, y_val_1), epochs=20, batch_size=batch_size, callbacks=[es, mc]).history
                else:
                    history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size).history
                is_best_model = True
                
            print("total number of chunks", number_of_chunks)
            filename = cls.h5_file_name +'.h5'
            path = os.path.join("..//gui_margarida//gui//assets", filename)
            model.save(path)
            print("Saved model to disk")
            
            X_test = np.array([0.2, 0.5, 0.67])
            X_test = X_test.reshape((1, cls.n_steps, cls.n_features))
            X_pred = model.predict(X_test)
        
            

            return True