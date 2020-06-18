# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:08:27 2020

@author: anama
"""

class Event:
    events = list()
    id_counter = 0
    def __init__(self, start, end, name, events=[], id_event=None):
        self.start = start
        self.end = end
        self.name = name
        if id_event == None:
            Event.id_counter += 1
            self.id = Event.id_counter
        else:
            self.id = id_event
            
        self.events = events
        self.avg_flow = None
        self.coef = None
        self.avgAnomalyScore = None
        
    def getId(self):
        return self.id
        
    def addEvent(self, event):
        self.append(event)
        
    def getName(self):
        return self.name
    
    def getStart(self):
        return self.start
    
    def getEnd(self):
        return self.end
    
    def setAvgFlow(self, average_flow):
        self.avg_flow = average_flow
        
    def setCoef(self, coef):
        self.coef = coef
        
    def setAvgAnomalyScore(self, avgAnomalyScore):
        self.avgAnomalyScore = avgAnomalyScore
        