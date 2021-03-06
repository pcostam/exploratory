# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:35:26 2020

@author: anama
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.series import create_data, series_to_supervised, generate_sequences, generate_normal
import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from keras.models import load_model
import base64
from io import BytesIO
import seaborn as sns
from preprocessing.seasonality import seasonal_adjustment, inverse_difference
from sklearn.model_selection import KFold
from scipy.stats import norm
from reconstruction_error import mean_abs_error, chebyshev, abs_error
from keras.models import model_from_json
import matplotlib.dates as mdates
    
def save_model_json(model, h5_filename):
    model_json = model.to_json()
    with open(os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", h5_filename + '_model.json'), "w") as json_file:
                json_file.write(model_json)
    model.save_weights(os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", h5_filename +'.h5'))
    print("Saved model to disk")
 
        
def load_model_json(h5_filename):
    model = model_from_json(open(os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", h5_filename + '_model.json')).read())
    model.load_weights(os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", h5_filename +'.h5'))
   
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
#input train
#output pred
def get_error_vector(y_true, y_pred, x_inv, scaler, timesteps, n_features, adjustment=False, metric="mae"): 
    """
    Inverses transform

    Parameters
    ----------
    x_input : TYPE
        DESCRIPTION.
    x_output : TYPE
        DESCRIPTION.
    timesteps : TYPE
        DESCRIPTION.
    n_features : TYPE
        DESCRIPTION.

    Returns
    -------
    vector : TYPE
        DESCRIPTION.

    """
    #inverse transform for forecasting
    y_pred = inverse_transform(y_pred, x_inv, scaler, adjustment=adjustment)
    y_true = inverse_transform(y_true, x_inv, scaler, adjustment=adjustment)
  
    y_true = process_predict(y_true, timesteps, n_features)
    y_pred = process_predict(y_pred, timesteps, n_features)   
    
    if metric == "mae":
        metric_func = mean_abs_error
    elif metric == "chebyshev":
        metric_func = chebyshev
    elif metric == "ae":
        metric_func = abs_error
        
    vector = metric_func(y_true, y_pred)
    print("vector", vector)
    vector = np.squeeze(vector)
    if len(vector.shape) == 1:
        vector = vector.reshape(vector.shape[0], 1)
   
    return vector
    
 # calculate anormaly score (X-mu)^Tsigma^(-1)(X-mu)
def anomaly_score(mu, sigma, X, n_features, type_score="ML"):
    # D- number of features
    #sigma DxD matrix
    #mu vector D-dimensional
    #X- error vector D-dimensional
    X = X.reshape(X.shape[0], n_features)
    if type_score == "ML":
        sigma_inv= np.linalg.inv(sigma)
        a = np.zeros(X.shape[0])
        for i in range(0, X.shape[0]):
                x = X[i, :]
                a[i] = (x - mu).T@sigma_inv@(x - mu)  
        return a

    elif type_score == "reconstruction error":
         return X
    
    

    

def process_input(X_train, y_train):
    if len(X_train.shape)==3:
        return X_train
          
    else:
        Xtrain = y_train
    return Xtrain


def get_threshold(dates, score):
    thresholds = [x for x in range(0, 20)]
    all_anormals = list()
    for th in thresholds:
        no_anomalous = 0
        i = 0
        for sc in score:
            if sc > th:
                no_anomalous += 1
            i += 1
        print("no_anomalous", no_anomalous)
        all_anormals.append(no_anomalous)
    all_anormals = np.array(all_anormals)
    index_min = np.argmin(all_anormals)
    min_th = thresholds[index_min]
    return min_th

def process_predict(X_pred,  timesteps, n_features): 
    print("n_features", n_features)
    print("timesteps", timesteps)
    print("X_pred shape 1", X_pred.shape)
    if len(X_pred.shape) == 3:
        X_pred = X_pred[:,:n_features,:]
        X_pred = X_pred.reshape(X_pred.shape[0], n_features)
        print("X_pred shape", X_pred.shape)
    else:
        X_pred = X_pred.reshape(X_pred.shape[0], n_features)
    print("X_pred.shape 4", X_pred.shape)
    return X_pred
def plot_training_losses(rep, history):
    fig, ax = plt.subplots(figsize=(14,6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mae)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    
    rep.add_image(fig, "Training losses")

    #plt.show()
    
    
def plot_bins_loss(title, x, rep, to_fit=None, mu=None, std=None, bins=20):
    figure, ax = plt.subplots(figsize=(16,9), dpi=80)
    plt.title(title, fontsize=16)
    sns.distplot(x, kde=True, color='blue')
 
    
    if to_fit == 'normal':
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2, color="red")
        
        
        
    #plt.show()
    """
    tmpfile = BytesIO()
    figure.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    """
    rep.add_image(figure, title) 
 

# Function that formats the axis labels
def timeTicks(x, pos):
    seconds = x / 10**9 # convert nanoseconds to seconds
    # create datetime object because its string representation is alright
    d = datetime.timedelta(seconds=seconds)
    return str(d)

def plot_series(title, dates, y_true, y_pred, timesteps, n_features, rep, x_inv, scaler, adjustment=False):
    #plot time series with real numbers
    #inverse transform for forecasting
    y_pred = inverse_transform(y_pred.reshape(-1, n_features), x_inv, scaler, adjustment=adjustment)
    y_true = inverse_transform(y_true.reshape(-1, n_features), x_inv, scaler, adjustment=adjustment)
    y_true = process_predict(y_true, timesteps, n_features)
    y_pred = process_predict(y_pred, timesteps, n_features)   
    
    df_true = pd.DataFrame()
    df_pred = pd.DataFrame()
    df_true['value'] = y_true.ravel()
    df_true.index = dates
    df_pred['value'] = y_pred.ravel()
    df_pred.index = dates
    
    print("df", type(df_pred.index))
    fig, ax = plt.subplots(figsize=(14,6), dpi=80)
    fig.autofmt_xdate()
    index_plot = df_true.index.values 
    plt.plot(index_plot, df_true['value'],'b', label="Validation")
    plt.plot(index_plot, df_pred['value'] , 'r', label="Prediction")
    from matplotlib import ticker
    formatter = ticker.FuncFormatter(timeTicks)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel('Value')
    ax.set_xlabel('Dates')
    ax.legend(loc='upper right')
    plt.show()
    rep.add_image(fig, title)


  
def concatenate_features(columns):
    agg = pd.concat(columns, axis=1, sort=False)
    agg =  agg.drop_duplicates("date", "first")
    col = (agg.columns == 'date').argmax()
    dates = agg.iloc[:, col]
    agg = agg.drop(['date'], axis=1)
    agg['date'] = dates                 
    return agg

def generate_full(raw, timesteps,input_form="3D", output_form="3D", n_seq=None, n_input=None, n_features=None):        
    X_train_full = preprocess(raw, timesteps, form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
 
    if output_form == "3D":
        y_train_full = X_train_full[:, :, :n_features]
    else:
        y_train_full = generate_full_y_train(raw, n_input, timesteps, n_features)
    return X_train_full, y_train_full



def preprocess(raw, timesteps, form="3D", input_data=pd.DataFrame(), n_seq=None, n_input=None, n_features=None, dates=False):   
    if isinstance(raw, pd.DataFrame) and dates == False:
        data = series_to_supervised(raw, n_in=n_input)
    else:
        data = raw
        
    
    data = preprocess_shapes(data, timesteps, form, input_data=input_data, n_seq=n_seq, n_input=n_input, n_features=n_features, dates=dates)
    
    return data

def fit_data(data, scaler):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    scaler : TYPE
        DESCRIPTION.

    """
    #normalize data
    if scaler == None:
        scaler = MinMaxScaler()
    scaler.partial_fit(data.values)
   
    return scaler

def fit_transform_data(data):
    scaler = MinMaxScaler()
    values = scaler.fit_transform(data.values)
    scaled = pd.DataFrame(values, index=data.index, columns=data.columns)
    return scaled
    

def transform_sequence(data, granularity='15min', lag='W', adjustment=True):
     print("granularity", granularity)
     if adjustment:
        data = seasonal_adjustment(data, granularity, lag=lag)
     return data
    
    
def transform_data(data, scaler):
    values = scaler.transform(data.values)
    scaled = pd.DataFrame(values, index=data.index, columns=data.columns)
    data = scaled
     
    return data

def inverse_transform(transformed, data, scaler, adjustment=True):
    inverted = scaler.inverse_transform(transformed)
    if adjustment:
        #difference_seasonality = len(data) - len(inverted)
        #inverted = inverted[difference_seasonality:]
        #print("inverted", len(inverted))
        inverted = np.array([inverse_difference(data[i], inverted[i]) for i in range(len(inverted))])
    return inverted 

def preprocess_shapes(data, timesteps, form="3D", input_data=pd.DataFrame(), n_seq=None, n_input=None, n_features=None, dates=False):
    if form == "4D":
        print("n_input", n_input)
        data = np.array(data.iloc[:, :n_input*n_features])
        print("data shape", data.shape)   
        #data = transform_data(data)
        #for CNN there is the need to reshape de 2-d np array to a 4-d np array [samples, timesteps, features]
        #[batch_size, height, width, depth]
        #[samples, subsequences, timesteps, features]
        columns = data.shape[1]
        samples = data.shape[0]
        cells = samples * columns
        new_n_seq = round(cells/(samples*timesteps*n_features))
        print("n_seq", n_seq)
        if new_n_seq != n_seq:
            print('Not possible to generate this number of sequences: %s. New sequence %s' %(n_seq, new_n_seq))
            n_seq = new_n_seq
        print("samples", samples)
        print("n_seq", n_seq)
        print("timesteps", timesteps)
        print("n_features", n_features)
        data = np.reshape(data, (samples, n_seq, timesteps, n_features))
    
        return data
       
    elif form == "2D":  
        data = data.values  
        y_train = data[:, :n_features]        
        #y_train = transform_data(y_train)    
        y_train =  np.squeeze(y_train)
        y_train = np.reshape(y_train, (y_train.shape[0], n_features))
     
        return y_train
    
    elif form == "3D":
        data = np.array(data.iloc[:, :timesteps*n_features])
        print("data", data.shape)
    
        #data = transform_data(data)

        #for lstm there is the need to reshape de 2-d np array to a 3-d np array [samples, timesteps, features]
        data = data.reshape((data.shape[0], timesteps, n_features))

        return data


def generate_full_y_train(normal_sequence, n_input, timesteps, n_features):
    data = series_to_supervised(normal_sequence, n_in=n_input)
   
    data = data.values
    
    if n_features == 1:
        y_train_full = [data[:, -1]]
    else:
        y_train_full = data[:, :n_features]
   
    
    #y_train_full = transform_data(y_train_full)
    
    y_train_full =  np.squeeze(y_train_full)
    
    return y_train_full


def generate_sets(raw, timesteps,input_form ="3D", output_form = "3D", validation=True, n_seq=None, n_input=None, n_features=None, dates=False, n_val_sets=2, train_split=0.8):       
    print(">>>>generate_sets")
    print("N_features", n_features)
    print("raw", len(raw))
    if validation:
        return generate_validation(raw, timesteps, input_form=input_form, output_form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features, dates=dates, n_val_sets=n_val_sets, train_split=train_split)

    X_train = preprocess(raw, timesteps, form = input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
  
    if output_form == "3D":
        y_train = X_train
    if output_form == "2D":
        print("2D")
        y_train = preprocess(raw, timesteps, form = output_form, input_data = X_train, n_seq=n_seq, n_input=n_input, n_features=n_features)
    print("generate_sets y_train shape", y_train.shape)
    print("generte_sets X_train shape", X_train.shape)
    return X_train, y_train, None, None, None, None
    

def split_features(n_features, X_train):
    print("split_features")
    print("n_features", n_features)
    input_data = list()
 
    for i in range(n_features):
        aux = X_train[:, :, :, i]
        reshaped = aux.reshape((aux.shape[0], aux.shape[1], aux.shape[2], 1))
        input_data.append(reshaped)
        
    return input_data



 
def generate_validation(X_train_D, timesteps,input_form="3D", output_form="3D",  n_seq=None, n_input=None, n_features=None, dates=False, n_val_sets=2, train_split=0.8):
     print("X_train_D")
     X_train = series_to_supervised(X_train_D, n_in=n_input, dates=dates)
             
     size_X_train = X_train.shape[0]
     size_train = round(size_X_train*train_split)
     X_val = X_train.iloc[size_train:, :]
     X_train = X_train.iloc[:size_train, :]
     size_val = round(0.5*X_val.shape[0])
     if n_val_sets == 2:
         X_val_1 = X_val.iloc[:size_val, :]
         X_val_2 = X_val.iloc[size_val:, :]
         
     elif n_val_sets == 1:
         X_val_1 = X_val
         X_val_2 = pd.DataFrame()  
         
     Xtrain = preprocess_shapes(X_train, timesteps,  form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     Xval_1 = preprocess_shapes(X_val_1, timesteps,  form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     if not(X_val_2.empty):
         Xval_2 = preprocess_shapes(X_val_2, timesteps,  form=input_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     else:
        Xval_2 = pd.DataFrame()         
         
     y_train = preprocess_shapes(X_train, timesteps, input_data = X_train, form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     y_val_1 = preprocess_shapes(X_val_1, timesteps, input_data = X_val_1, form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     if not(X_val_2.empty):
         y_val_2 = preprocess_shapes(X_val_2, timesteps, input_data = X_val_2, form=output_form, n_seq=n_seq, n_input=n_input, n_features=n_features)
     else:
         y_val_2 = None
     print("X_val_1", X_val_1.shape)
     print("y_val_1", y_val_1.shape)
     print("type", type(Xval_1))
   
     return Xtrain, y_train, Xval_1, y_val_1, Xval_2, y_val_2
 
def save_parameters(scaler, mu, sigma, timesteps, th_min, th_max, filename, fpr, tpr):
    print("save parameters")
    param = {'mu':mu, 'sigma':sigma, 'timesteps':timesteps, 'th_min': th_min, 'th_max': th_max, 'scaler': scaler, 'fpr': fpr, 'tpr': tpr}
    filename = 'parameters_' + filename + '.pickle'
    path = os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", filename)
    with open(path, 'wb') as f:  
        pickle.dump(param, f, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def load_parameters(filename):
    global mu, sigma, timesteps, th_min, th_max
    filename = 'parameters_' + filename + '.pickle'
    path = os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", filename)
 
    # Getting back the objects:
    with open(path, 'rb') as f:  
        param = pickle.load(f)
    
    mu = param['mu']
    sigma = param['sigma']
    timesteps = param['timesteps']
    th_min = param['th_min']
    th_max = param['th_max']
    
    return param

def save_train_parameters(filename, normal_sequence, anormal_sequence, sequence_inv, scaler, history, split):
    print("save parameters")
    param = {'normal sequence': normal_sequence, 'anormal sequence': anormal_sequence, 'sequence': sequence_inv, 'scaler': scaler, 'history': history, 'split':split}
    filename = 'trainparameters_' + filename + '.pickle'
    path = os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", filename)
    with open(path, 'wb') as f:  
        pickle.dump(param, f, protocol=pickle.HIGHEST_PROTOCOL)
    return True

def load_train_parameters(filename):
    filename = 'trainparameters_' + filename + '.pickle'
    path = os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", filename)
 
    # Getting back the objects:
    with open(path, 'rb') as f:  
        param = pickle.load(f)
    

    return param


def save_cv_parameters(filename, run_losses, run_val_losses):
    print("save parameters")
    param = {'run losses': run_losses, 'run val losses': run_val_losses}
    filename = 'traincvparameters_' + filename + '.pickle'
    path = os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", filename)
    with open(path, 'wb') as f:  
        pickle.dump(param, f, protocol=pickle.HIGHEST_PROTOCOL)
    return True
def load_cv_parameters(filename):
    filename = 'traincvparameters_' + filename + '.pickle'
    path = os.path.join(os.getcwd(), "wisdom/gui_margarida/gui/assets", filename)
 
    # Getting back the objects:
    with open(path, 'rb') as f:  
        param = pickle.load(f)
    

    return param

def get_date(df):
    return df['date']


def generate_days(X_train_D, n_input, time, validation=True, n_val_sets=2, train_split=0.8):   
    X_train = pd.DataFrame()
    X_train = series_to_supervised(X_train_D, n_in=n_input, dates=True)
    if validation == True:   
       
        size_X_train = X_train.shape[0]
       
        size_train = round(size_X_train*train_split)
        X_val = X_train.iloc[size_train:, :]
        X_train = X_train.iloc[:size_train, :]
        
        if n_val_sets == 2:
             size_val = round(0.5*X_val.shape[0])
             X_val_1 = X_val.iloc[:size_val, :]
             X_val_2 = X_val.iloc[size_val:, :]
             
        elif n_val_sets == 1:
             X_val_1 = X_val
             X_val_2 = pd.DataFrame()
           
      
    
        if isinstance(X_train.index, pd.TimedeltaIndex):
            print("ISISTANCE>>>>>>>>>>>>>>>>>>")
            Xval2 = None
            if not(X_val_2.empty):
                Xval2 = X_val_2.index.to_pytimedelta()
            return X_train.index.to_pytimedelta(), X_val_1.index.to_pytimedelta(), Xval2 
        else:
            Xval2 = None
            if not(X_val_2.empty):
                Xval2 = X_val_2.index.values
            return X_train.index.values, X_val_1.index.values, Xval2
    else:
        if isinstance(X_train.index, pd.TimedeltaIndex):
            return X_train.index.to_pytimedelta()
        else:
            return X_train.index.values

def drop_date(df):
    df = df.drop(['date'], axis = 1)
    return df


def make_score(X_test, y_test, y_inv, h5_filename, timesteps, n_features, th_min=None, th_max=None, time='date', metric="chebyshev"):   
    model = load_model_json(h5_filename) 
    
    print("Predict")
    y_pred = model.predict(X_test)
    
    param = load_parameters(h5_filename)
    scaler = param['scaler']
    print("y_test['value']", y_test['value'].shape)
    ####### INVERSE TRANSFORM
    vector = get_error_vector(y_test['value'].values.reshape(-1,1), y_pred, y_inv, scaler, timesteps, n_features, metric=metric)
    vector = np.squeeze(vector)
        
    score = anomaly_score(mu, sigma, vector, n_features)
    return score

def make_prediction(score, X_test, y_test, h5_filename, timesteps, n_features, th_min, th_max, time='date'):
    values = list()
    dates = list()
    mask = list()
    predict = pd.DataFrame()
    anomalies = 0
    i = 0
    
    param = load_parameters(h5_filename)
    mu = param['mu']
    sigma = param['sigma']
    timesteps = param['timesteps']
    scaler = param['scaler']
    print("mu", mu)
    print("sigma", sigma)
    print("timesteps", timesteps)
     
    if th_min == None:
         th_min = param['th_min']
    if th_max == None:
         th_max = param['th_max']
         
    for sc in score:
        date = y_test.index.values[i]
        value = y_test['value'].values[i]
        dates.append(date)
        values.append(value)
        if th_min == None:
            if sc > th_max:
                 anomalies += 1
                 mask.append(True)
            else:
                 mask.append(False)
        elif th_min != None:
            if sc > th_max or sc < th_min:
                 anomalies += 1
                 mask.append(True)
            else:
                 mask.append(False)
            
        i += 1
    print("Number of anomalies in detect_anomalies", anomalies)
 
    predict['value'] = values
    predict['is_anomaly'] = mask
    if time == 'date':
        dates = pd.to_datetime(dates)
   
    predict.index = dates
    print("end")
    print("rows predict", predict.shape[0])
    return predict

def detect_anomalies(X_test, y_test, y_inv, h5_filename, timesteps, n_features, th_min, th_max, time='date', metric="chebyshev"):    
    score = make_score(X_test, y_test, y_inv, h5_filename, timesteps, n_features, th_min=th_min, th_max=th_max, time=time, metric=metric)
    predict = make_prediction(score, X_test, y_test, h5_filename, timesteps, n_features, th_min=th_min, th_max=th_max, time=time)
    return score, predict

def avgAnomalyScore(score, min_th, network, X_test_D):
     i = 0
     events = network.getEvents()
     curr_id_event = None
     anomaly_scores = list()
     for sc in score:
        date = X_test_D.index.values[i]
        for event in events:
            if date >= event.getStart() and date <= event.getEnd():
                id_event = event.getId()
                #there is a new event
                if curr_id_event != id_event:
                    event = network.getEventById(curr_id_event)
                    event.setAvgAnomalyScore(np.mean(np.array(anomaly_scores)))  
                    curr_id_event = id_event
                    anomaly_scores = list()
                #its the same event    
                anomaly_scores.append(X_test_D['value'].iloc[i]) 
        i += 1
    
    
def join_partitions_features(train_chunks_all, no_partitions_cv,to_exclude):
    """Concatenates information of several sensors"""
    train_chunks = list()
    for i in range(0,no_partitions_cv):
        to_concat = [item[i] for item in train_chunks_all] 
        union_normal_sequences = pd.concat(to_concat, axis=1) 
        union_normal_sequences =  union_normal_sequences.drop_duplicates(to_exclude, "first")
        for column in to_exclude:
            col = (union_normal_sequences.columns == column).argmax()
            dates = union_normal_sequences.iloc[:, col]
            union_normal_sequences = union_normal_sequences.drop([column], axis=1)
            union_normal_sequences[column] = dates                 
        train_chunks.append(union_normal_sequences)
    return train_chunks
          

def test_2d():
    col_1 = [x for x in range(20)]
    col_2 = [x for x in range(20)]
    d = {'value': col_1, 'date':col_2}
    df = pd.DataFrame(data=d)
    y_test = preprocess(df, 2, form="2D", input_data=pd.DataFrame(), n_seq=2, n_input=4, n_features=1)
    print("y_test shape", y_test.shape)
    #comeu 4 porque e' o n_input e o n_input nao considera os nans
    return True

def test():
   col_1 = [x for x in range(20)]
   col_2 = [x for x in range(20)]
   col_3 = [x for x in range(20,40)]
   col_4 = [x for x in range(20)]
  
   d = {'value': col_1, 'date':col_2}
   d2 = {'value': col_3, 'date': col_4}
   df = pd.DataFrame(data=d)
   df2 = pd.DataFrame(data=d2)
   
   df_union = pd.concat([df, df2], axis=1)
   col = (df_union.columns == 'date').argmax()
   dates = df_union.iloc[:, col]
   df_other = df_union.drop(['date'], axis=1)
   df_other['date'] = dates
   print("dates", dates)
   #df_other['date'] = dates
   print("df.columns", df_union.columns)
   print("type", type(df_union.columns))
   print(df.columns.where(df.columns == 'date'))
  
   return df_other
   
def split_folds(data, n_folds=3):
    rows = data.shape[0]
    split = round(rows/3)
    res = []
    first = 0
    pivot = 0
    while pivot <= rows:
        pivot += split
        res.append(data[first:pivot])
        first = pivot
     
    for fold in res:
        print("fold shape", fold.shape)

    return res


def convert_time_rows(time, unit):
    return True    