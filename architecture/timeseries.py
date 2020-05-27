# -*- coding: utf-8 -*-
"""
Created on Tue May 26 17:50:17 2020

@author: anama
"""
import matplotlib.pyplot as plt
class timeseries(object):
    
    def __init__(self, name, dates, y_true, y_pred):
        self.name = name
        self.dates = dates
        self.y_true = y_true
        self.y_pred = y_pred
        
    def do_histogram(X):
        plt.hist(list(X), bins=20)
        plt.show()
        
    def get_name(self):
        return self.name
    
    def get_dates(self):
        return self.dates
    
    def get_y_true(self):
        return self.y_true
    
    def get_y_pred(self):
        return self.y_pred