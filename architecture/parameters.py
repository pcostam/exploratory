# -*- coding: utf-8 -*-
"""
Created on Wed May 27 12:04:19 2020

@author: anama
"""

class parameters(object):
   
    
    def __init__(self, n_features=1, dropout=False, n_steps=96, n_input=None,  n_seq=None, regularizer="L1"):
        self.n_features = n_features
        self.n_input = n_input
        self.dropout = dropout
        self.n_steps = n_steps
        self.n_seq = n_seq
        self.regularizer = regularizer
        
        if n_seq != None:
            self.n_input = self.n_steps*self.n_seq
      
        
    def get_n_features(self):
        return self.n_features
    
    def get_n_input(self):
        return self.n_input
    
    def get_dropout(self):
        return self.dropout
    
    def get_n_seq(self):
        return self.n_seq
    
    def get_regularizer(self):
        return self.regularizer
    
    def get_n_steps(self):
        return self.n_steps
    
    def get_simulated(self):
        return self.simulated
    
    def get_n_train(self):
        return self.n_train
    
    def set_n_steps(self, n_steps):
        self.n_steps = n_steps
        
    def set_n_features(self, n_features):
        self.n_features = n_features
        
    