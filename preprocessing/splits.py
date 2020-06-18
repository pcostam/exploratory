# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:27:03 2020

@author: anama
"""

from base.Network import Network
import pandas as pd

def convert_time(no_months):
    minutes = 15*4*24*30*no_months
    rows = int(round(minutes/15))
    return rows
    

def expanding_window(normal_sequence, all_sequence, n_train, events, test_split=0.2, gap=0, time="date", blocked=True):
    """
    

    Parameters
    ----------
    normal_sequence : TYPE
        DESCRIPTION.
    all_sequence : TYPE
        DESCRIPTION.
    n_train : minimum size train.
    test_split : TYPE, optional
        DESCRIPTION. The default is 0.2.
    gap : TYPE, optional
        DESCRIPTION. The default is 0.
    blocked : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    train_chunks : TYPE
        DESCRIPTION.
    test_chunks : TYPE
        DESCRIPTION.

    """
    #print("normal_sequence", normal_sequence.index)
    #print("all_sequence", all_sequence.index)
    n_train = convert_time(3)
    n_test = convert_time(1)
    #n_test = get_no_instances_test(n_train, test_split)
    train_chunks = list()
    test_chunks = list()
    n_records = all_sequence.shape[0]
    #n_train = n_train
    print("n_records", n_records)

    while True:
        train = normal_sequence
        train = train[:n_train]
     
        #not enough samples to make train sequence
        if train.shape[0] < n_train or train.empty:
            break
        
      
        #Date that should begin anomalous sequence for test sequence
        end_date = train.index.values[-1]
        print("end date", end_date)
        test = all_sequence[(all_sequence.index > end_date)]
        test = test[:n_test]   
        
        #not enough samples to make test sequence
        if test.shape[0] < n_test or test.empty:
            print("n_test", n_test)
            break
        
        if time == 'date':
            test.index = pd.to_datetime(test.index)
            train.index = pd.to_datetime(train.index)
  
        
        
        """
        if not(Network.findSubSequenceWithEvent(train, events, time).empty):
               raise ValueError("Normal Sequence with Events")
        """
        
        if not(Network.findSubSequenceWithEvent(test, events, time).empty):
                   print(">>>>TRAIN", train.index)
                   print(">>>>TEST", test.index)
                   test_chunks.append(test)
                   train_chunks.append(train)
             
        n_train += n_test
         
    
         

    
    return train_chunks, test_chunks