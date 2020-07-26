# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 16:16:44 2020

@author: anama
"""
from scipy.stats import norm
from kstest import goodness_of_fit
from utils import plot_bins_loss
from report import Text
from architecture.utils import detect_anomalies, make_prediction, make_score
from preprocessing.series import change_format
from evaluation.metrics import confusion_matrix, f1_score, precision, accuracy, recall, getTP, getFP, getFN, getTN, f_beta_score
import numpy as np

    
def test_betas(X_anormal, y_anormal_D, h5_file_name, n_steps, n_features, events, file, time='date',  choose_th=None, beta=0.1, type_sequence='normal'):
        ths = np.arange(0.1, 2.0, 0.1)
        betas = np.arange(0.01, 0.9, 0.01)   
        score, _ = make_score(X_anormal, y_anormal_D, h5_file_name, n_steps, n_features, time=time)
        print("score", score[:15])
        import os
      
        for b in betas:
            fbs_max = None
            th_max = 0
            param = dict()
            for th_to_test in ths:
                prn, acc, th_max, fbs_max, param = metrics_th(score, param, b, th_max, fbs_max, X_anormal, y_anormal_D, h5_file_name, n_steps, n_features, events, file, th_to_test, time='date', type_sequence=type_sequence)
                
            if fbs_max == None:
                param['beta'] = b
                param['fbs_max'] = fbs_max
            print("Beta:", b)
            print("Maximum threshold chosen fbetascore", th_max )
            print("Maximum fbetascore", fbs_max)
      
            path = os.path.join(os.getcwd(), "wisdom/architecture/","report_betas.txt" )
            
            
            if os.path.exists(path):
                append_write = 'a' # append if already exists
            else:
                append_write = 'w' # make a new file if not
    
            with open(path, append_write) as f:
                print("write file test number 4")
                f.write("%s \n" % (param))
                f.close()       
        return True
    

