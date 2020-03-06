# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:38:28 2020

@author: anama
"""

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from scipy.stats import norm, combine_pvalues

import numpy as np

class Pvalue:
    
    pvalues = {"box-cox":[],"yeo-john":[], "None":[]}
    
    def getPvalues(self, transform="None"):
        return self.pvalues[transform]
        
    #kernel density estimation
    #see https://machinelearningmastery.com/probability-density-estimation/
    #and https://scikit-learn.org/stable/modules/density.html
    #and https://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html
    ##https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
    def KDE(sample):
        #reshape data to [rows,columns]
        sample = np.array(sample)
        sample = sample.reshape((len(sample), 1))
        
        
        # use grid search cross-validation to optimize the bandwidth
        #params = {'bandwidth': np.logspace(-1, 1, 20)}
        params = {'bandwidth': 10 ** np.linspace(-1, 1, 100)}
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), params, n_jobs = -1)
        
        grid.fit(sample)
        
        model = grid.best_estimator_
        
        #save into file
        joblib.dump(model, 'KDE.pkl')
       
        return model
    
    #see https://machinelearningmastery.com/critical-values-for-statistical-hypothesis-testing/
    # tails of distribution https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html
    def pvalue_norm(mu, std, percentage=0.05):
        #critical values from ppf at 1%, 5% and 10%  
        crit = norm.ppf(1-percentage, loc=mu, scale=std)
       
        #right-tailed test
        #pvalue = 1 - norm.cdf(c, loc=mu, scale=std)
        #two-tailed test
        pvalue = 2*(1 - norm.cdf(crit, loc=mu, scale=std))
           
        print("p_value is ", pvalue)
        
        return crit, pvalue
            
    def consensus(pvalues, crits, alfa=0.05, transform ="None"):
        #need to be pi/2 on stouffers methods
        pvalues = [x / 2 for x in pvalues]
        
        
        stat, pvalue = combine_pvalues(pvalues, method="stouffer")
        print("pvalues all:", str(pvalues))
        print("stat:" + str(stat) + "pvalue" + str(pvalue))
         
        if pvalue < alfa:
            #point failure, anomaly in time 
            return False
        else:
            #common event, spatial anomaly
            return True
            
            
            
            
        
        
