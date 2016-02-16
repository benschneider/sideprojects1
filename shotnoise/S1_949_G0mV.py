# -*- coding: utf-8 -*-
'''
@author: benschneider
A script is used to readout mtx measurement data which also contains
a shotnoise responses.
Then fits them for G and Tn
'''

import numpy as np
from parsers import savemtx, loadmtx, make_header
from scipy.optimize import curve_fit  # , leastsq
# import scipy.optimize
from scipy.constants import Boltzmann as Kb
from scipy.constants import h, e  # , pi
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit  # , Parameter


def xderiv(d2MAT, dx=1.0, axis=0):
    '''
    This  derivative is inaccurate at the edges.
    Calculates a 3p derivative of a 1D, 2D matrix.
    This does not require you to shift the xaxis by one half pt.
    dx = distance between points
    '''
    if len(IVs.shape) > 1:
        if axis == 1:
            ''' Not tested yet could be faster than a matrix transpose'''
            a2 = np.zeros([d2MAT.shape[0]+2, d2MAT.shape[1]])
            a2[1:-1, :] = d2MAT
            m1 = d2MAT - a2[:-2, :]
            m2 = a2[2:, :] - d2MAT
            dy = (m1+m2)/2.0
            dy[0, :] = dy[1, :]
            dy[-1, :] = dy[-2, :]
        elif axis == 0:
            a2 = np.zeros([d2MAT.shape[0], d2MAT.shape[1]+2])
            a2[:, 1:-1] = d2MAT
            m1 = d2MAT - a2[:, :-2]
            m2 = a2[:, 2:] - d2MAT
            dy = (m1+m2)/2.0
            dy[:, 0] = dy[:, 1]
            dy[:, -1] = dy[:, -2]
        return dy/dx
    else:
        a2 = np.zeros([d2MAT.shape[0]+2])
        a2[1:-1] = d2MAT
        m1 = d2MAT - a2[:-2]
        m2 = a2[2:] - d2MAT
        dy = (m1+m2)/2.0
        dy[0] = dy[1]
        dy[-1] = dy[-2]
        return dy/dx


def find_nearest(someArray, value):
    ''' This function helps to find the index corresponding to a value
    in an array.
    Usage: indexZero = find_nearest(myarray, 0.0)
    returns: abs(myarray-value).argmin()
    '''
    idx = abs(someArray-value).argmin()
    return idx


def fitfunc(I, R, G, Tn, T):
    # def fitfunc(x, G, Tn, T, c):
    '''
    This contains the fitting equation, which i use to fit the
    shot noise response.
    returns: fit-value(x, ...)
    '''
    E1 = (e*I*R+h*f)/(2*Kb*T)
    E2 = (e*I*R-h*f)/(2*Kb*T)
    Si = ((2*Kb*T/R) * (E1/np.tanh(E1) + E2/np.tanh(E2)))
    return (B*G*(Si * R**2 + 4.0*Kb*T*R + 4.0*Kb*Tn*Z0 *
                 (R**2+Zopt*Zopt)/(Z0*Z0+Zopt*Zopt)) *
            (Z0/((R+Z0)*(R+Z0))))


'''
Loading the data files I1I1, Q1Q1, I2I2, Q2Q2, Vm
d1, d2, d3 are all the same since they all originate from the same type of
measurement.
'''

filein1 = 'S1_949_G0mV_SN_PCovMat_cI1I1'
filein2 = 'S1_949_G0mV_SN_PCovMat_cQ1Q1'
filein3 = 'S1_949_G0mV_SN_PCovMat_cI2I2'
filein4 = 'S1_949_G0mV_SN_PCovMat_cQ2Q2'
filein5 = 'S1_949_G0mV_SN_PV_2'
I1I1, d3, d2, d1, dz = loadmtx('sn_data//' + filein1 + '.mtx')
Q1Q1, d3, d2, d1, dz = loadmtx('sn_data//' + filein2 + '.mtx')
I2I2, d3, d2, d1, dz = loadmtx('sn_data//' + filein3 + '.mtx')
Q2Q2, d3, d2, d1, dz = loadmtx('sn_data//' + filein4 + '.mtx')
Vm,   dv3, dv2, dv1, dvz = loadmtx('sn_data//' + filein5 + '.mtx')

lags0 = find_nearest(d1.lin, 0.0)  # lags position
PD1 = (I1I1[lags0]+Q1Q1[lags0])
PD2 = (I2I2[lags0]+Q2Q2[lags0])

print np.mean(PD1[0])/np.mean(PD2[0])  # i.e. 0.779 ~ 28% difference

# Getting the differential Resistance
# ------------------------------------
# And Optionally filter discontinuities
# RTR = 1009.1                  # Ib Resistance in kOhm
# RG = 1000.0                   # Pre Amp gain factor
# dv3.scale = 1e3/(RTR)         # scale to uAmp
# dv3.update_lin()
# IVs = IVs * 1e6/RG            # scale to uVolts
# the axis and data scaling was done beforehand for this data set
d3 = dv3                        # update all d3 axis
pidx = 0                        # power index 0 is 0 power
IVs = Vm[0, pidx]
dx3 = dv3.lin[1] - dv3.lin[0]   # get step-size for the derivative
dIV = xderiv(IVs, dx3)
# dIVm = np.mean(dIV, axis=0)   # this was used to simply average several traces
dR = abs(dIV)                  # no filter other than an absolute val for now
dRm = gaussian_filter1d(dR, 1.5)  # Gaussian filter
# dRm = abs(dRm)                  # no filter other than an absolute val for now
# ------------------------------------


Z0 = 50.0
Zopt = 50.0
B = 1e5
f = 4100*1e6
I = d3.lin*1e-6  # in Amps

plt.ion()
crop_within = find_nearest(I, -0.9e-6), find_nearest(I, 1.1e-6)
# crop_within = find_nearest(I, -7.3e-6), find_nearest(I, 7.3e-6)
crop_outside = find_nearest(I, -19.5e-6), find_nearest(I, 19.5e-6)
crop = [crop_within, crop_outside]
# crop_within = find_nearest(I, -4e-6), find_nearest(I, 4e-6)
# crop = find_nearest(I, -2e-6), find_nearest(I, 2e-6), 2.3e-10, 5.6e-10
crop2 = [crop_within, crop_outside]


data = PD1[0]*1.0
params = Parameters()
params.add('Tn', value=3.7, vary=True, min=2, max=4)
params.add('G', value=3.38e7, vary=True, min=1e6, max=1e9)
params.add('T', value=0.012, vary=False)


def fitfun2(params, I, dRm):
    G = params['G'].value
    Tn = params['Tn'].value
    T = params['T'].value
    return fitfunc(I, dRm, G, Tn, T)


def ministuff(params, I, dRm, measd, crop):
    SNfit = fitfun2(params, I, dRm)
    # crop data for fitting:
    # crop vertically
    # SNfit[SNfit < crop[2]] = 0
    # SNfit[SNfit > crop[3]] = 0
    # measd[measd < crop[2]] = 0
    # measd[measd > crop[3]] = 0
    # crop horizontally anything too close to zero
    SNfit[crop[0][0]:crop[0][1]] = 0
    measd[crop[0][0]:crop[0][1]] = 0
    SNfit[0:(crop[1][0])] = 0
    SNfit[crop[1][1]:-1] = 0
    measd[0:(crop[1][0])] = 0
    measd[crop[1][1]:-1] = 0
    return (measd-SNfit)*1e10


result = minimize(ministuff, params, args=(I, dRm, data*1.0, crop))
print report_fit(result)
deltaTn = result.params['Tn'].stderr
deltaG = result.params['G'].stderr
SNfit = fitfun2(result.params, I, dRm)



