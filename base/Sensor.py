# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 18:25:23 2020

@author: anama
"""

class Sensor:
    def __init__(self, id_sensor, neighbors, sensor_type, network_type, pvalues_neighbors=dict(), pvalue_self=None):
        #int
        self.id_sensor = id_sensor
        #list
        self.neighbors = neighbors
        #dict key:value, where key is sensor id and value is pvalue
        self.pvalues_neighbors = pvalues_neighbors
        #float
        self.pvalue_self = pvalue_self
        #string either pressure or flow
        self.sensor_type = sensor_type
        #string either tm ou tg
        self.network_type = network_type
    
    def getId(self):
        return self.id
    def getNeighbors(self):
        return self.neighbors
    def getPvaluesNeighbors(self):
        return self.pvalues_neighbors
    def getPvaluedSelf(self):
        return self.pvalue_self
    def getSensorType(self):
        return self.sensor_type
    def getNetworkType(self):
        return self.network_type
    
    def setPvalueSelf(self, pvalue):
        self.pvalue_self = pvalue
    
    def setPvaluesNeighbors(self, pvalues_neighbors):
        self.pvalues_neighbors = pvalues_neighbors
        
    