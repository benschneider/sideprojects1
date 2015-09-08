# -*- coding: utf-8 -*-
"""
Created on Tue May  6 16:15:47 2014

@author: benschneider
"""
import numpy as np
def norm_1D_filter(filterfun):
    '''
    go across each element and adds them
    this value is then subtracted from the filter function
    (N = integral{(-inf) to (+inf)}{function})
    '''
    N = 0
    for i in range(0,len(filterfun)):
        N += filterfun[i]

    return filterfun/N

def convolution_1d(data2d, filt_fun):
    tmp = data2d
    for j in range(0,len(tmp)):
        tmp[j] = _convolution(tmp[j], filt_fun)
    return tmp

def _convolution(data, filt_func):
    '''
    This function does calculate a 1D convolution
    ,the goal is to test a convolution method.
    (it is not yet optimized for speed)
    expected input: func is the data to be processed 
    and filt_func is the filter to be used
    '''
    length = len(data)
    filt_new = _get_convolution_filter(filt_func,length)
    #new_filt =  [0,0,0,0,0,0,0.5,1,0.5,0,0,0,0,0,0]
    filt_length = len(filt_func)
    
    #dim1 = (len(data)+filt_length)
    
    dim2 = len(data)
    test_array = np.zeros([dim2,dim2])

    for i in range(0,dim2):
        #for each element in data do
        filt_start = length+filt_length/2-i
        filt_stop  = (2*length+filt_length/2-i)
        filt_window = filt_new[filt_start:filt_stop]
        #print filt_window        
        test_array[i] = filt_window*data[i]
    
    data_convolved = test_array.sum(0) 
    return data_convolved
    
def _get_convolution_filter(filt_func,length):
    '''puts the filter in the center and adds sufficient zeros at either side'''
    #length = len(filt_func)
    a = np.array([0]*length)
    b = np.array([0]*(length+1))    
    new_filt = np.concatenate((a,filt_func,b), axis = 0)
    return new_filt