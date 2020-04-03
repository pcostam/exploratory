# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:02:33 2020

@author: anama
"""

def finess():
    print("hello world")
    
class Apple(object):
    my_func = finess  
    
    @classmethod
    def get_func(cls):
        return Apple.my_func
    
    @classmethod
    def do_stuff(cls):
        my_func = Apple.get_func()
        my_func()
        
        
#Apple.do_stuff()