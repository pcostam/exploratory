# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:00:09 2020

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
    
class son(father):
    c = "another one"
    y="i am your son"
    #son is not defined
    #b = son.x
    #este ja resulta
    b = father.x
    x = 4
  
    
    @classmethod
    def banana(cls):
        print(son.coisas())
    
    def aux():
        son.banana()
    #son not defined
    #a = son.coisas()
    #not defined
    #a = coisas()
    a = father.coisas()

#https://stackoverflow.com/questions/11058686/various-errors-in-code-that-tries-to-call-classmethods
#son.banana()
#son.coisas()
#son.c
#son.y
#son.x
#son.wayToDo()
#son.anotherway()