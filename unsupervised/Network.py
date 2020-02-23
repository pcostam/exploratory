# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 12:41:53 2020

@author: anama
"""
import mysql.connector
from Event import Event
import datetime
class Network:
  
    sensors = list()
    events = list()
    def __init__(self, name):
        self.name = name
 
    def getEvents(self):
        return self.events
    
    def addEvent(self, event):
        self.events.append(event)
        
    def addAllEvents(self, db, cursor):
        query = """
        SELECT descricao, date FROM ordemdata ORDER BY date ASC
        """
        cursor.execute(query)
        
        aux = list(cursor)
     
        size_aux = len(aux)
        for i in range(0,size_aux):
            descricao = aux[i][0]
            date = aux[i][1]
            if descricao == 'Percepcao':
                try:
                    #fuga
                    date = aux[i-1][1]
                    start = Network.date_N_days_ago(date, 2)
                    #end vai ser quando for abertura
                    end = aux[i+1][1]
                    j = Network.find_open(i, aux, "Abertura")
                    abertura = Event(aux[j][1], aux[j][1], aux[j][0], []) 
                    events = [abertura]
                    self.addEvent(Event(start, end, "fuga", events))
                    """
                    else:
                        #ocorreu apenas fecho. (reparacao?) nao sei se e' fuga
                        Event(date, date,"reparo", [])
                        Network.find_open(i, aux)
                    """
                        
                except IndexError as error:
                    # Output expected IndexErrors.
                    pass

    def date_N_days_ago(date, days):
        return date - datetime.timedelta(days=days) 

    def find_open(init, aux, name_event):
         j = init
         while aux[j][0] != name_event:
             j += 1
         
         return j
    
    def countEvent(self):
        size = len(self.events)    
        return size
    
    def findEventType(self, type_event, date):
        size = len(self.events)
        for i in range(0, size):
            start = self.events[i].getStart()
            end = self.events[i].getEnd()
            if date >= start and date <= end and self.events[i].getName == type_event:
                return self.events[i]
        return None
     
    def findEvent(self, date):
        size = len(self.events)
        for i in range(0, size):
            start = self.events[i].getStart()
            end = self.events[i].getEnd()
            if date >= start and date <= end:
                return self.events[i]
        return None
    
    #TODO
    def plot_timeline_events(self):
        return True
        
		     
                