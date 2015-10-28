import numpy as np
from parsers import savemtx, loadmtx, make_header
from scipy.optimize import curve_fit  # , leastsq
# import scipy.optimize
from scipy.constants import Boltzmann as Kb
from scipy.constants import h, e  # , pi
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

# Start values
Z0 = 50.0
Zopt = 50.0
T = 0.012
R = 75.0
Tn = 3.1
B = 1e6
f = 5800*1e6
G = 59627515
RTR = 1009.1  # Resistance at RT in kOhm in front of V output

folder = 'mtx_out//'
filein = 'S1_631_SN_G100_BPF4'
M1, d1, d2, d3, dz = loadmtx('mtx_out//' + filein + '.mtx')
# data = loadmtx('mtx_out//' + filein + '.mtx')
M1 = data[0]


d1.scale = 1000.0/RTR  # scale to uAmp
d1.update_lin()
IVs = M1[8]
IVs = IVs * 1000.0  # scale to uVolts
PD1 = M1[0]
PD2 = M1[1]


def xderiv(d2MAT, dx=1.0, axis=0):
    '''
    This  derivative is inaccurate as the edges.
    Calculates a 3p derivative of a 2D matrix.
    This does not require you to shift the xaxis by one half pt.
    dx = distance between points
    '''
    if axis == 1:
        ''' Not tested yet should be faster than a matrix transpose'''
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

# dIV containts the averaged differential Resistance
dx1 = d1.lin[1] - d1.lin[0]
dIV = xderiv(IVs, dx1)
dIVm = np.mean(dIV, axis=0)
# Calculate a Gausian filter across it to rid of some noise
dRm = gaussian_filter1d(dIVm, 5)  # 1d array, num of points to filter

plt.close()
plt.figure(1)
plt.plot(d1.lin, dIVm)


def find_nearest(someArray, value):
    idx = abs(someArray-value).argmin()
    return idx


def Rm(x, c=0):
    if c == 1:
        return np.abs(dRm2[x])
    if c == 0:
        return np.abs(dRm[x])


def fitfunc(x, G, Tn, T, c):
    R = Rm(x, c)
    I = d1.lin[x]*1e-6  # because needs to be in Amps here
    E1 = (e*I*R+h*f)/(2*Kb*T)
    E2 = (e*I*R-h*f)/(2*Kb*T)
    Si = ((2*Kb*T/R) * (E1/np.tanh(E1) + E2/np.tanh(E2)))
    return (B*G*(Si * R**2 + 4.0*Kb*T*R + 4.0*Kb*Tn*Z0
                 * (R**2+Zopt*Zopt)/(Z0*Z0+Zopt*Zopt))
            * (Z0/((R+Z0)*(R+Z0))))


def fitfun2(x, G, Tn, c=1):
    T = 0.012
    return fitfunc(x, G, Tn, T, c)*1e12

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
savemtx('mtx_out//' + filein + 'test.mtx', MAT1, header=header1)
