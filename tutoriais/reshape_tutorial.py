# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 15:18:13 2020

@author: anama
"""
import numpy as np
x = np.array([[[1,2,3,4,5], [4,5,6,7,8]],
                [[2,4,6,8,12],[3,6,9,12,15]],
                [[3,8,8,9,8], [9,9,10,11,19]]])

print("x shape", x.shape)
no_elements = x.shape[0]*x.shape[1]*x.shape[2]

print("no elements", no_elements)

#I divided in for shape[1]
for i in range(x.shape[1]):
    aux = x[:, i]
    print("aux", aux)
    print("aux shape", aux.shape)
    print("no elements", aux.shape[0] * aux.shape[1])
    reshaped = aux.reshape((-1, x.shape[0], 1))
    print("reshaped", reshaped)
    print("reshaped", reshaped.shape)
#input_x = [input_x[:,i].reshape((1,input_x.shape[0],1)) for i in range(input_x.shape[1])]
    
    

#4-d array
    
y = np.array([[[[ 3,  30],
                [ 8,  1]],
    
                [[ 5, 10],
                [15, 20]]],
    
            [[[ 6, 12],
              [18, 24]],

            [[ 7, 14],
             [21, 28]]]])
    
    
print("y shape", y.shape)

for i in range(y.shape[3]):
    aux = y[:, i]
    print("aux", aux)
    print("aux shape", aux.shape)
    print("no elements", aux.shape[0] * aux.shape[1])
    reshaped = aux.reshape((aux.shape[0], aux.shape[1], aux.shape[2], 1))
    print("reshaped", reshaped)
    print("reshaped", reshaped.shape)
