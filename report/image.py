# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:53:33 2020

@author: anama
"""
#leaf class
class Image:
    def __init__(self, title, encoded):
        self.title = title
        self.encoded = encoded
    
    def getTitle(self):
        return self.title
    
    def getEncoded(self):
        return self.encoded
    
    def writeToHtml(self): 
        html_string = """
        <div>
        """ + self.getTitle() + """<img src=\'data:image/png;base64,{}\'>'""".format(self.getEncoded()) +"""
        </div>
        """
        return html_string