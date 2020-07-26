# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:27:03 2020

@author: anama
"""

from base.Network import Network
import pandas as pd

def convert_time(no_months, time):
    rows = 0
    if time == 'date':
        minutes = 4*24*30*no_months
        rows = int(round(minutes))
    elif time == 'time':
        minutes = 6*24*30*no_months
        rows = int(round(minutes))
        
    return rows
    

def expanding_window(normal_sequence, all_sequence, n_train, n_test, events, test_split=0.2, gap=0, time="date", blocked=True):
    """
    

    Parameters
    ----------
    normal_sequence : TYPE
        DESCRIPTION.
    all_sequence : TYPE
        DESCRIPTION.
    n_train : minimum size train. Units months
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
    if events == []:
        raise ValueError("Expanding window has no events")
    #print("normal_sequence", normal_sequence.index)
    #print("all_sequence", all_sequence.index)
    n_train = convert_time(n_train,time)
    n_test = int(round(n_train*test_split))
    #n_test = convert_time(n_test,time)
    print(">>>n_test", n_test)
    print(">>>n_train", n_train)
    train_chunks = list()
    test_chunks = list()
    n_records = all_sequence.shape[0]
    print("n_records", n_records)

    while True:
        train = normal_sequence
        train = train[:n_train]
        print("len train", len(train))
        print("tran.shape", train.shape[0])
        print("N-train", n_train)
  
        #not enough samples to make train sequence
        if train.shape[0] < n_train or train.empty:
            print("not enough train samples")
            break
        
      
        #Date that should begin anomalous sequence for test sequence
        end_date = train.index.values[-1]
        print("end date", end_date)
        test = all_sequence[(all_sequence.index > end_date)]
        test = test[:n_test]   
        
        #not enough samples to make test sequence
        if test.shape[0] < n_test or test.empty:
            print("not enough test samples")
            break
        
        if time == 'date':
            test.index = pd.to_datetime(test.index)
            train.index = pd.to_datetime(train.index)
  
        
        
        """
        if not(Network.findSubSequenceWithEvent(train, events, time).empty):
               raise ValueError("Normal Sequence with Events")
        """
        print("OTHER TIME", time)
        if not(Network.findSubSequenceWithEvent(test, events, time).empty):
                   print(">>>>TRAIN", train.index)
                   print(">>>>TEST", test.index)
                   test_chunks.append(test)
                   train_chunks.append(train)
             
        n_train += n_test
        #n_test = int(round(n_train*test_split))
         
    
         
    print("len-trainchunks", len(train_chunks))
    
    return train_chunks, test_chunks


def get_no_instances_test(n_train, test_split=0.2):
    """ 
    Returns number where of necessary instances to make the 
    percentage test split necessary
    Parameters
    ----------
    n_train : int
        number of train instances
    test_split : float, optional
        percentage of test sequence
    
    """
    return int(round((test_split*n_train)/(1-test_split)))


def split_train_test(normal_sequence, all_sequence, n_train, test_split=0.2, gap=0, time="date", blocked=True):
    #rolling out
    """
    Splits sequence into normal sequence(to train) and test sequence
    (normal and anomalous), making into chunks.
    
    Parameters
    ----------
    Dataframe
    n_train : int
    Size of each chunk.
    
    Returns
    -------
    None.

    """
    n_test = get_no_instances_test(n_train, test_split)
    print("n_train", n_train)
    print("n_test", n_test)
    margin = 0
    train_chunks = list()
    test_chunks = list()
    n_records = len(all_sequence)
    start_date = normal_sequence.index.values[0]
    
    while margin < n_records:
        train = normal_sequence[(normal_sequence.index >= start_date)]
        if train.empty:
            break
        train = train[:n_train]
        #not enough samples to make train sequence
        if train.shape[0] < n_train:
            break
        #Date that should begin anomalous sequence for test sequence
        start_date = train.index.values[-1]
        test = all_sequence[(all_sequence.index > start_date)]
        test = test[:n_test]   
        
        #not enough samples to make test sequence
        if test.shape[0] < n_test:
            break
        #Date that should end anomalous sequence for test sequence
        #and begin new train chunk
        end_date = test.index.values[-1]
        start_date = end_date 
        #+ datetime.timedelta(days=gap)
        print("train shape", train.shape)
        print("train.index", train.index)
        print("test shape", test.shape)
        print("test.index", test.index)
        test_chunks.append(test)
        train_chunks.append(train)
     
        if blocked:
            margin += n_train + n_test -1
        else:
            margin += n_train - 1
    
    return train_chunks, test_chunks

#blocked margin = n_train + n_test
def rolling_out_cv(X, n_train, test_split=0.2, gap=0, blocked=True):
    """ With X, divides in train and test chunks """
    margin = 0
    train_chunks = list()
    test_chunks = list()
    n_records = len(X)
    start = 0
    
    n_test = round((test_split*n_train)/(1-test_split))
    print("n_val", n_test)

    i = 0
    while margin < n_records:
        start = i + margin
        i += 1
        stop = start + n_train
        train = X[start:stop]
        
        #index test set
        start = stop + gap
        stop = start + n_test
        
        if X[start:stop].empty:
            break

        test_chunks.append(X[start:stop])
        train_chunks.append(train)
        if blocked:
            margin += n_train + n_test -1
        else:
            margin += n_train - 1
    print("train_chunks", len(train_chunks[0])) 
    print("test_chunks", len(test_chunks[0])) 
    return train_chunks, test_chunks