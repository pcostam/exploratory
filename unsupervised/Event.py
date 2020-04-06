# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:08:27 2020

@author: anama
"""

class Event:
    events = list()
    id_counter = 0
    def __init__(self, start, end, name, events):
        self.start = start
        self.end = end
        self.name = name
        Event.id_counter += 1
        self.id = Event.id_counter
        self.events = events
        
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