# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:41:22 2020

@author: anama
"""
from matplotlib import pyplot
from preprocessing.series import create_dataset
from Pvalue import KDE
import numpy as np
class Sensor:
    def __init__(self, name):
        self.name = name
        self.instant = -1
        self.all_diff = list()
        self.sort_diff = list()
 
        
    def update(self, actual_measure, prev_measure, neighbors_measures):
        self.instant += 1
      
        diff = [round(prev_measure - actual_measure, 2)]
        for i in range(0, len(neighbors_measures)):
            diff.append(round(actual_measure - neighbors_measures[i-1],2))
            
        self.all_diff.append(diff)
        self.sort_diff += diff
        self.sort_diff.sort()
       
    
    
    def plot_diff_measure(sample, probabilities):
        #https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
        pyplot.hist(sample, bins=50)
        pyplot.plot(sample, probabilities)
        pyplot.show()
        
        
    def test():
        df = create_dataset("sensortgmeasure", "12")
        value = list(df["value"])
        print("value", len(value))
        
        neighbor_1 = create_dataset("sensortgmeasure", "10")
        neighbor_1_val = neighbor_1["value"]
        neighbor_2 = create_dataset("sensortgmeasure", "1")
        neighbor_2_val = neighbor_2["value"]
        
        sensor = Sensor("RPR Caudal Grv")
        
        for i in range(0,len(df)):
            try:
                sensor.update(value[i], value[i-1], [neighbor_1_val[i], neighbor_2_val[i]])
            except IndexError:
                sensor.update(value[i], 0, [neighbor_1_val[i], neighbor_2_val[i]])
        sample = sensor.sort_diff
        print("all_diff", sample)
       
        sample = np.array(sample)
        sample = sample.reshape((len(sample), 1))
        #density function
        model = KDE(sample)
        
        probabilities = model.score_samples(sample)
        
        Sensor.plot_diff_measure(sample, probabilities)
        
            
        
    
   