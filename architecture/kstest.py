# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:43:27 2020

@author: anama
"""
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from report import Text

def goodness_of_fit(file, scores, alpha=0.05):
    distributions = ['norm', 'gamma']
    results = []
    for dist in distributions:
        k_stat, ks_pvalue = 0, 0
        if dist == 'gamma':
            k_stat, ks_pvalue = kstest(scores, 'gamma', args=(15.5, 0, 1./7))
        else:
            ks_stat, ks_pvalue = kstest(scores, dist)
        
        info = dict()
        info["stat_D"] = ks_stat
        info["p-value"] = ks_pvalue
        info["distribution"] = dist
        results.append(info)
              
        if ks_pvalue >= alpha:
            text ="Distribution is: %s" % dist
            print(text)
            file.append(Text.Text(text))
        else:
            text = "Failed test to distribution %s" % dist
            print(text)
            file.append(Text.Text(text))
            
        stats = [item["stat_D"] for item in results]
        p_value = [item["p-value"] for item in results]
        distribution = [item["distribution"] for item in results]
        minimum_stat = stats[0]
        p_value_min = p_value[0]
        dist_min = distribution[0]
        for i in range(1, len(stats)):
            actual = stats[i]
            if minimum_stat > actual:
                minimum_stat = actual
                p_value_min = p_value[i]
                dist_min = distribution[i]
        text = """Minimum distance is %s for the distribution %s
        with p-value of %s""" % (minimum_stat, dist_min, p_value_min)
        file.append(Text.Text(text))
       
def ecdf(data):
    fig = plt.figure()
    x = np.sort(data)
    n = len(x)
    y = np.arange(1, n+1) / n
    plt.plot(x, y, marker='.', linestyle='none')
    plt.xlabel('Losses')
    plt.ylabel('Quantile')
    plt.margins(0.02)
    plt.show()

    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
     
    return encoded


def transform_norm(data, type_transform="yeojohnson"):
    mu, std = 0 
    if type_transform == "box-cox":
        #box-cox transformation
        print("Minimum value:" , min(data))
        
        shift = 0
        minimum = min(sample)
        if minimum < 0:
            shift = round(abs(minimum))
     
        posdata = [x + shift for x in sample] 
        
        posdata, lmda = boxcox(posdata)
      
        mu, std = norm.fit(posdata)
   
    elif type_transform == "yeojohnson":
        #Yeo-Johnson power transformation
        datayeo, lmbda = yeojohnson(data)
        mu, std = norm.fit(datayeo)
        
    return mu, std


