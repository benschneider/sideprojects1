# -*- coding: utf-8 -*-
'''
@author: benschneider
A script used to readout mtx measurement data which contains
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


def Rm(x, c=True):
    ''' This function is used to handle the differential resistance values
    since a differential resistance can be negative or infinite at single point
    jumps. This definition requires two globals which contain the differential
    resistance. dRm and dRm2.. I used this to test different ways of calculating
    and obtaining differential resistances.
    '''
    if c is True:
        return np.abs(dRm[x])


def fitfunc(x, G, Tn, T, c):
    '''
    This contains the fitting equation, which i use to fit the
    shot noise response.
    returns: fit-value(x, ...)
    '''
    R = Rm(x, c)
    I = d1.lin[x]*1e-6  # because needs to be in Amps here
    E1 = (e*I*R+h*f)/(2*Kb*T)
    E2 = (e*I*R-h*f)/(2*Kb*T)
    Si = ((2*Kb*T/R) * (E1/np.tanh(E1) + E2/np.tanh(E2)))
    return (B*G*(Si * R**2 + 4.0*Kb*T*R + 4.0*Kb*Tn*Z0 *
                 (R**2+Zopt*Zopt)/(Z0*Z0+Zopt*Zopt)) *
            (Z0/((R+Z0)*(R+Z0))))


def fitfun2(x, G, Tn, c=True):
    T = 0.012
    return fitfunc(x, G, Tn, T, c)*1e12


'''
Loading all the data files I1I1, Q1Q1, I2I2, Q2Q2, Vm
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

# Roughly the difference between the two is roughly:
print np.mean(PD1[0])/np.mean(PD2[0])
# i.e. 0.779 ~ 28% difference

# Getting the differential Resistance
# ------------------------------------
# First get the units right,
# Current in uA and voltage in uV
# Then take the derivative
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
# dRm = gaussian_filter1d(dIV, 5)  # 5pt Gaussian filter avoid discontinuities
dRm = abs(dIV)                  # no filter other than an absolute val for now
# ------------------------------------


#  Plot data loaded so far
plt.ion()
plt.figure(1)
plt.imshow(PD1, aspect='auto')
plt.figure(2)
plt.plot(PD1[0])
plt.figure(3)
plt.plot(d3.lin, abs(dRm))
plt.show()

# Fitting start parameters
Z0 = 50.0
Zopt = 50.0
T = 0.012
R = 75.0
Tn = 3.1
B = 1e6
f = 5800*1e6
G = 59627515


'''
To obtain the differential resistance we need to take the derivative
of the measured voltage with respect to its current.
to take this derivative we use xderiv function :
'''

'''
# initguess = [G, Tn, T]
initguess1 = [G, Tn]
initguess2 = [G, Tn]


# crop data to be fitted
start1 = 200
stop1 = 800
start2 = 1600
stop2 = 2300
dRm2 = dRm
xdata2 = np.concatenate((range(start1, stop1), range(start2, stop2)), axis=0)
PD12 = np.concatenate((PD2[:, start1:stop1], PD1[:, start2:stop2]), axis=1)
PD22 = np.concatenate((PD2[:, start1:stop1], PD2[:, start2:stop2]), axis=1)
numN = PD12.shape[0]
G1G2Tn1Tn2 = np.zeros([numN, 4])

i = 0
for i in range(numN):
    ydata1 = PD12[i]*1e12
    ydata2 = PD22[i]*1e12
    popt, pcov = curve_fit(fitfun2, xdata2, ydata1, p0=initguess1)
    popt2, pcov2 = curve_fit(fitfun2, xdata2, ydata2, p0=initguess2)
    initguess1 = popt  # for the next position
    initguess2 = popt2
    G1G2Tn1Tn2[i, 0], G1G2Tn1Tn2[i, 1] = popt
    G1G2Tn1Tn2[i, 2], G1G2Tn1Tn2[i, 3] = popt2

# def func(params) = ydata - fitfun2(xdata, params)
# scipy.optimize.leastsq(

ydata = PD1[50]*1e12
xdata = range(PD1.shape[1])
y2 = fitfun2(xdata, G1G2Tn1Tn2[50, 0], G1G2Tn1Tn2[50, 1], c=0)
error = ydata - y2
plt.figure(2)
plt.plot(d1.lin, ydata)
plt.plot(d1.lin, y2)


MAT1 = np.zeros([1, G1G2Tn1Tn2.shape[0], G1G2Tn1Tn2.shape[1]])
MAT1[0, :, :] = G1G2Tn1Tn2
header1 = make_header(d1, d2, d3, meas_data=('Gain'))
savemtx('mtx_out//' + filein1 + 'test.mtx', MAT1, header=header1)

'''
