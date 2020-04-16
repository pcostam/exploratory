# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:30:29 2020

@author: anama
"""


from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from keras.models import Sequential
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import tensorflow
from keras.backend import clear_session
from keras import regularizers
from keras.optimizers import Adam
import time
from architecture import utils
from architecture.EncDec import EncDec
from preprocessing.series import generate_sequences, generate_normal
from keras.callbacks import EarlyStopping
from architecture import tuning
#see https://towardsdatascience.com/lstm-autoencoder-for-anomaly-detection-e1f4f2ee7ccf
#https://towardsdatascience.com/lstm-autoencoder-for-extreme-rare-event-classification-in-keras-ce209a224cfb
#see https://medium.com/@crawftv/parameter-hyperparameter-tuning-with-bayesian-optimization-7acf42d348e1
#see https://medium.com/@shivajbd/understanding-input-and-output-shape-in-lstm-keras-c501ee95c65e
#see https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html#skopt.gp_minimize
#https://www.kdnuggets.com/2019/06/automate-hyperparameter-optimization.html



class autoencoderLSTM(EncDec):
    input_form = "3D"
    output_form = "3D"
    config = []
    print("testing class")
    @classmethod
    def get_input_form(cls):
        return cls.input_form
    @classmethod
    def get_output_form(cls):
        return cls.output_form
    
    def autoencoder(X, y, config): 
        print("config", config)
        print("toIndex", EncDec.toIndex)
        toIndex = EncDec.toIndex
        num_lstm_layers =  tuning.get_param(config, toIndex, "num_lstm_layers")
        num_lstm_layers_compress = tuning.get_param(config, toIndex, "num_lstm_layers_compress")
        learning_rate = tuning.get_param(config, toIndex, "learning_rate") 
        drop_rate_1 =  tuning.get_param(config, toIndex, "drop_rate_1")
     
        drop_rate_2 =  tuning.get_param(config, toIndex, "drop_rate_2")
              
        
        print("X", X)
        timesteps = EncDec.n_steps
        n_features = EncDec.n_features
        model = Sequential()
        # Encoder  
        model.add(LSTM(64, activation='relu', 
                       input_shape=(timesteps, n_features), 
                       return_sequences=True))
        for i in range(num_lstm_layers):
            name = 'layer_lstm_encoder_{0}'.format(i+1)
            model.add(LSTM(64, activation='relu', return_sequences=True, 
                           name=name)) 
            
        model.add(LSTM(32, activation='relu',return_sequences=False))
        for i in range(num_lstm_layers_compress):
               model.add(LSTM(32, activation='relu',return_sequences=False))
        model.add(Dropout(rate=drop_rate_1))
        model.add(RepeatVector(timesteps))
        # Decoder
        model.add(LSTM(32, activation='relu', return_sequences=True))
        for i in range(num_lstm_layers_compress):
            name = 'layer_lstm_decoder_compress_{0}'.format(i+1)
            model.add(LSTM(32, activation='relu', return_sequences=True, name=name))
        for i in range(num_lstm_layers):
             model.add(LSTM(64, activation='relu', 
                            return_sequences=True)) 
            
        model.add(LSTM(64, activation='relu', return_sequences=True)) 
        model.add(Dropout(rate=drop_rate_2))
        model.add(TimeDistributed(Dense(n_features)))
     
        adam = Adam(lr=learning_rate)
        model.compile(optimizer=adam, loss='mae')
        model.summary()
        
        return model


    type_model_func = autoencoder
   
    
    def hyperparam_opt():
        dim_num_lstm_layers = Integer(low=0, high=20, name='num_lstm_layers')
        dim_num_lstm_layers_compress = Integer(low=0, high=20, name="num_lstm_layers_compress")
        dim_batch_size = Integer(low=64, high=128, name='batch_size')
        #dim_adam_decay = Real(low=1e-6,high=1e-2,name="adam_decay")
        dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
        dim_drop_rate_1 = Real(low=0.2 ,high=0.9,name="drop_rate_1")
        dim_drop_rate_2 = Real(low=0.2 ,high=0.9,name="drop_rate_2")
        dimensions = [dim_num_lstm_layers,
                      dim_batch_size,
                      dim_learning_rate,
                      dim_drop_rate_1,
                      dim_drop_rate_2,
                      dim_num_lstm_layers_compress]
        
        default_parameters = [0, 128, 1e-2, 0.5, 0.5, 0]
        
        for i in range(0, len(dimensions)):
             EncDec.toIndex[dimensions[i].name] = i
        
        return dimensions,  default_parameters
    
    dimensions,  default_parameters = hyperparam_opt()
    config = default_parameters
    #regularized?
    #config += [True]
    @use_named_args(dimensions=dimensions)
    def fitness(num_lstm_layers, batch_size, learning_rate, drop_rate_1, drop_rate_2):  
        init = time.perf_counter()
        print("fitness>>>")
        print("number lstm layers:", num_lstm_layers)
        print("batch size:", batch_size)
        print("learning rate:", learning_rate)
        print("drop rate 1:", drop_rate_1)
        print("drop rate 2:", drop_rate_2)
    
        normal_sequence, _ = generate_sequences("12", "sensortgmeasurepp", df_to_csv=True, start=EncDec.stime, end=EncDec.etime)
        print("stime", autoencoderLSTM.stime)
        print("etime", autoencoderLSTM.etime)
        X_train_full, _ = utils.generate_full(normal_sequence,autoencoderLSTM.n_steps, input_form=autoencoderLSTM.input_form, output_form=autoencoderLSTM.output_form, n_seq=autoencoderLSTM.n_seq,n_input=autoencoderLSTM.n_input, n_features=autoencoderLSTM.n_features)
        
        config = [num_lstm_layers, batch_size, learning_rate,  drop_rate_1, drop_rate_2]
        model = autoencoderLSTM.type_model_func(X_train_full, X_train_full, config)
    
        print("total number of chunks", len(normal_sequence))
        no_chunks = 0
        for df_chunk in normal_sequence:
            no_chunks += 1
            print("number of chunks:", no_chunks)
            X_train, y_train,  X_val_1, y_val_1, X_val_2, y_val_2 = utils.generate_sets(df_chunk, 96, input_form=autoencoderLSTM.get_input_form(), output_form=autoencoderLSTM.get_output_form()) 
            es = EarlyStopping(monitor='val_loss', min_delta = 0.01, mode='min', verbose=1)
            hist = model.fit(X_train, X_train, validation_data=(X_val_1, y_val_1), epochs=100, batch_size= batch_size, callbacks=[es])
            
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

    mu = 0
    sigma = 0
    min_th = 0
    h5_file_name = "autoencoderLSTM"
    fitness_func = fitness
    
    
  
        


