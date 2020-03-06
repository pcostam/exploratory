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
    def writeToHtml(self, filename):
        try:
            
            string_html = str()
            for element in self.elements:
                string_html += element.writeToHtml()
            self.writeFile(string_html, filename)
           
            return string_html
        except AttributeError:
            print('The operation cant be done', element)
    
    def writeFile(self, string_html, name):
         print("WRITE FILE")
         filename = "F:/Tese/exploratory/wisdom/reports_files/report_pvalue/%s.html" % name
         self.name = filename
         print("filename", filename)
         if not os.path.exists(os.path.dirname(filename)):
             try:
                os.makedirs(os.path.dirname(filename))
             except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

         with open(filename, "w") as f:
             f.write(string_html)
             f.close()

    def append(self, htmlContent):
        self.elements.append(htmlContent)
        
    def getElements(self):
        return self.elements
        
   