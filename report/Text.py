# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:57:46 2020

@author: anama
"""

class Text:
     def __init__(self, text, tag="p"):
         self.tag = tag
         self.text = text
         
     def writeToHtml (self):
        html_string = "<%s>" % self.tag
        html_string = self.text
        html_string += "</%s>" % self.tag
        return html_string