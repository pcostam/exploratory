# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 20:21:43 2019

@author: anama
"""
import matplotlib.pyplot as plt

def visualize(nr_parts, X_train, y_pred, yi, size_x_train, dba_km):
    for yi in range(nr_parts):
        plt.subplot(nr_parts, nr_parts, yi + 1)
        for xx in X_train[y_pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, size_x_train)
        plt.ylim(-4, 4)
        if yi == 1:
            plt.title("DBA $k$-means")