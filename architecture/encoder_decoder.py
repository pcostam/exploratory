# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:12:48 2020

@author: anama
"""
from EncDec import EncDec
class encoder_decoder(EncDec):   
    def __init__(self, model_name, type_model):
        if model_name == "CNN-BiLSTM":
            encoder_decoder.encoder = "CNN"
            encoder_decoder.decoder = "BiLSTM"