# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:38:28 2020

@author: anama
"""

from sklearn.neighbors import KernelDensity
import numpy as np

#kernel density estimation
#see https://machinelearningmastery.com/probability-density-estimation/
#and https://scikit-learn.org/stable/modules/density.html
def KDE(sample):
    model = KernelDensity(bandwidth=2, kernel='gaussian')
    #reshape data to [rows,columns]
    sample = np.array(sample)
    sample = sample.reshape((len(sample), 1))
    model.fit(sample)
    return model