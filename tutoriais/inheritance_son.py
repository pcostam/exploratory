# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:12:39 2020

@author: anama
"""

from tutoriais.inheritance_father import father
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