# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:29:39 2020

@author: anama
"""
import os
import errno

#Works as component class but is composite class
class HtmlFile:
    
    elements = list()
 
    #this makes that the agreement imposed by an "interface" is met
    def writeToHtml(self, path):
        try:
            
            string_html = str()
            for element in self.elements:
                string_html += element.writeToHtml()
            self.writeFile(string_html, path)
           
            return string_html
        except AttributeError:
            print('The operation cant be done', element)
    
    def writeFile(self, string_html, path):
         print("WRITE FILE")
         self.name = path
         print("filename", path)
         if not os.path.exists(os.path.dirname(path)):
             try:
                os.makedirs(os.path.dirname(path))
             except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

         with open(path, "w") as f:
             f.write(string_html)
             f.close()

    def append(self, htmlContent):
        self.elements.append(htmlContent)
        
    def getElements(self):
        return self.elements
        
   