# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:12:30 2020

@author: anama
"""

class father(object):
    x = 2
    y = "oi"
    
    

    @classmethod
    def coisas(cls):
        return cls.x
    
    
    def wayToDo():
        return father.coisas() + 6
    
    @classmethod
    def anotherway(cls):
        return cls.coisas() + 6
    