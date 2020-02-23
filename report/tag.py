# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:34:50 2020

@author: anama
"""
from report.HtmlElement import HtmlElement as Element
class Body(Element):
    def __init__(self): 
        self.tag = "body"

class Head(Element):
     def __init__(self):
         self.tag = "head"
    
class Div(Element):
     def __init__(self):
         self.tag = "div"
    
class Html(Element):
    def __init__(self):
        self.tag = "html"
            
class Style(Element):
    
    tag = "style"
    
    
               
        
        
    