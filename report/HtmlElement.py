# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:33:34 2020

@author: anama
"""
#Composite class
class HtmlElement:
    elements = []
  
    def writeToHtml (self):
        html_string = "<%s>" % self.tag
        for element in self.elements:
            html_string += element.writeToHtml()
        html_string += "</%s>" % self.tag
        return html_string
    
    
    def append(self, element):
        self.elements.append(element)