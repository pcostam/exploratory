# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:11:04 2020

@author: anama
"""
from architecture.parameters import parameters
class user_parameters(parameters):
    def __init__(self, n_features=1, dropout=False, n_steps=96, n_input=None,  n_seq=None, regularizer="L1", n_train=None
                 , simulated = False, validation=True, bayesian=False, save=False, hidden_size=16, code_size=4, n_leaks=500):
        
        super().__init__(n_features=n_features
                         ,dropout = dropout
                         ,n_steps = n_steps
                         ,n_seq = n_seq
                         ,regularizer = regularizer)
        
        self.n_train = n_train
        self.simulated = simulated
        self.save = save
        self.bayesian = bayesian
        self.validation = validation
        self.hidden_size = hidden_size
        self.code_size = code_size
        self.n_leaks = n_leaks
    def get_simulated(self):
        return self.simulated
    def get_n_leaks(self):
        return self.n_leaks
    def get_hidden_size(self):
        return self.hidden_size
    def get_code_size(self):
        return self.code_size
    def get_bayesian(self):
        return self.bayesian
    def get_n_train(self):
        return self.n_train
    def get_validation(self):
        return self.validation
    def get_save(self):
        return self.save
          
        

    