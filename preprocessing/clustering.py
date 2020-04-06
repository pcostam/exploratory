# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 19:25:03 2019

@author: anama
"""

from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
X = random_walks(n_ts=50, sz=32, d=1)
km = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=5,random_state=0).fit(X)
print(km.cluster_centers_.shape)