# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:48:45 2020

@author: anama
"""
#https://www.python-course.eu/passing_arguments.php
#asteric in function calls
def lol(*x):
    print(*x)

p = (47,11,12)
lol(*p)
#(47, 11, 12)

#double asteric in function calls
def f(a,b,x,y):
    print(a,b,x,y)
    
d = {'a':'append', 'b':'block','x':'extract','y':'yes'}
f(**d)

#('append', 'block', 'extract', 'yes')