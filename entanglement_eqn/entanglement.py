# -*- coding: utf-8 -*-
'''
@author: Ben Schneider

Basically extract degree of entanglement from the data files
'''
import numpy as np
import SNfit

cv = SNfit.my_variables_class()
cv.filein1 = 'S1_949_G0mV_SN_PCovMat_cI1I1.mtx'
cv.filein2 = 'S1_949_G0mV_SN_PCovMat_cQ1Q1.mtx'
cv.filein3 = 'S1_949_G0mV_SN_PCovMat_cI2I2.mtx'
cv.filein4 = 'S1_949_G0mV_SN_PCovMat_cQ2Q2.mtx'
cv.filein5 = 'S1_949_G0mV_SN_PV.mtx'
cv.filein6 = 'S1_949_G0mV_SN_PCovMat_cI1I2.mtx'
cv.filein7 = 'S1_949_G0mV_SN_PCovMat_cI1Q2.mtx'
cv.filein8 = 'S1_949_G0mV_SN_PCovMat_cQ1I2.mtx'
cv.filein9 = 'S1_949_G0mV_SN_PCovMat_cQ1Q2.mtx'
cv.fifolder = 'sn_data//'
cv.load_and_go()

SNdata = SNfit.DoSNfits



def get_LogNegNum(CovM):
    '''
    function to calculate Log-Negativity from a covariance matrix

    CovM looks a bit like:
        I1  I1I1 I1Q1 I1I2 I1Q2
        Q1  Q1I1 Q1Q1 Q1I2 Q1Q2
        I2  I2I1 I2Q1 I2I2 I2Q2
        Q2  I2Q1 Q1Q1 I2Q2 Q2Q2

            I1   Q1   I2   Q2
    '''
    V = np.linalg.det(CovM)
    A = np.linalg.det(CovM[:2, :2])
    B = np.linalg.det(CovM[2:, 2:])
    C = np.linalg.det(CovM[:2, 2:])
    s = A + B - 2.0*C
    vn = np.sqrt(s/2.0 - np.sqrt(s**2.0 - 4.0*V) / 2)
    return -np.log10(2.0*vn)


def TwoModeSqueeze_inequality(f1, f2, n):
    ''' This function calculates the amount of squeezing required
    to prove entanglement
    '''
    return ((2.0*np.sqrt(f1*f2)*(2*n))
            / (f1*(2*n+1)+f2*(2*n+1)))



