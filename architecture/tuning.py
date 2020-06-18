# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:50:42 2020

@author: anama
"""
from skopt.space import Integer, Real
from skopt.callbacks import DeltaYStopper
from keras.backend import clear_session
from skopt import gp_minimize
from skopt import Optimizer
from joblib import Parallel, delayed

#https://medium.com/@crawftv/parameter-hyperparameter-tuning-with-bayesian-optimization-7acf42d348e1
#https://www.kdnuggets.com/2019/06/automate-hyperparameter-optimization.html
#num_pooling_layers, stride_size, kernel_size, no_filters
def get_param_conv_layers(timesteps, n_input):
    dim_num_pooling_layers = Integer(low=0, high=2, name='num_pooling_layers')
    dim_stride_size = Integer(low=1, high=2, name='stride_size')
    dim_kernel_size = Integer(low=3, high=5, name='kernel_size')
    dim_no_filters = Integer(low=3, high=20, name='no_filters')
    
    dimensions = [dim_num_pooling_layers, dim_stride_size, dim_kernel_size, dim_no_filters]
    default_parameters = [1, 2, 5, 16]
    
    return dimensions, default_parameters

#num_encdec_layers, batch_size, learning_rate, drop_rate_1
def get_param_encdec(timesteps):
    dim_encdec_layers = Integer(low=0, high=2, name='num_encdec_layers')
    dim_batch_size = Integer(low=64, high=timesteps, name='batch_size')
    dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform', name='learning_rate')
    dim_drop_rate_1 = Real(low=0.2 ,high=0.9,name="drop_rate_1")
    dimensions = [dim_encdec_layers,
                  dim_batch_size,
                  dim_learning_rate,
                  dim_drop_rate_1]
    
    default_parameters = [2, 64, 1e-2, 0.25]
    
    return dimensions, default_parameters


def verifyConditions(parameters, timesteps, n_seq):
    print("verifyConditions")
    num_pooling_layers = parameters[0] 
    stride_size = parameters[1]
    kernel_size = parameters[2] 
    no_filters = parameters[3]
    num_encdec_layers = parameters[4]
    batch_size = parameters[5] 
    learning_rate = parameters[6] 
    drop_rate_1 = parameters[7]
    
    print("kernel_size:", kernel_size)
   
    if (kernel_size % 2) == 0:
        print("False 1")
        return False
    if batch_size > timesteps:
        print("False 2")
        return False
    if kernel_size > (timesteps * n_seq):
        print("False 3")
        return False
    
    #num_encdec_layers?
    
    print("True")
    return True

def do_bayesian_optimization(fitness, dimensions, default_parameters, timesteps, n_seq):
    print("START BAYESIAN OPTIMIZATION")
    print("fitness", fitness)
    print("dimensions", len(dimensions))
    print("default parameters", len(default_parameters))
    param = []
    
    opt = Optimizer(dimensions=dimensions,
                    acq_func='EIps')
    n_calls = 11
    i = 0
    res = []
    while i != n_calls:
        next_x = list()
        if i == 0:
            next_x = default_parameters
        else:
            next_x = opt.ask()
            
        while verifyConditions(next_x, timesteps, n_seq)==False:
            next_x = opt.ask()
            
        if verifyConditions(next_x, timesteps, n_seq):
            f_val = fitness(next_x) 
            res = opt.tell(next_x, f_val)
            i += 1
       
    
    clear_session()
    return res
            
        
    
    """
    es = DeltaYStopper(0.01)
    
    gp_result = gp_minimize(func=fitness,
                                dimensions=dimensions,
                                n_calls=11,
                                noise= 0.01,
                                n_jobs=-1,
                                x0=default_parameters,
                                callback=es, 
                                random_state=12,
                                acq_func="EIps")
    print("END BAYESIAN OPTIMIZATION")
    param = gp_result.x     
    clear_session()
    """
    
    
    
    return param 



#get parameters
def get_param(dimensions, toIndex, name):
    return dimensions[toIndex[name]]
