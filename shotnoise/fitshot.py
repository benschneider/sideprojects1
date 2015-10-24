import numpy as np
from parsers import savemtx, loadmtx
from scipy.optimize import curve_fit, leastsq
from scipy.constants import Boltzmann as Kb
from scipy.constants import h, e  # , pi
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

d1.scale = 1000.0/RTR  # scale to uAmp
d1.update_lin()
IVs = M1[8]
IVs = IVs * 1000.0  # scale to uVolts
PD1 = M1[0]
PD2 = M1[1]

# 200, 2300
# crop data to be fitted
d1.lin2 = d1.lin[200:2300]
d1.pt = 2300-200
IVs2 = IVs[:, 200:2300]
PD12 = PD1[:, 200:2300]
PD22 = PD2[:, 200:2300]

d1.lin = d1.lin2
IVs = IVs2
PD1 = PD12
PD2 = PD22


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
plt.close()
plt.figure(1)
plt.plot(d1.lin, dIVm)


def find_nearest(someArray, value):
    idx = abs(someArray-value).argmin()
    return idx


def Rm(x):
    return np.abs(dIVm[x])


def fitfunc(x, G, Tn, T):
    R = Rm(x)
    I = d1.lin[x]*1e-6  # because needs to be in Amps here
    E1 = (e*I*R+h*f)/(2*Kb*T)
    E2 = (e*I*R-h*f)/(2*Kb*T)
    Si = ((2*Kb*T/R) * (E1/np.tanh(E1) + E2/np.tanh(E2)))
    return (B*G*(Si * Rm(x)**2 + 4.0*Kb*T*Rm(x)
                 + 4.0*Kb*Tn*Z0*(Rm(x)**2 + Zopt*Zopt)/(Z0*Z0 + Zopt*Zopt))
            * (Z0/((Rm(x)+Z0)*(Rm(x)+Z0))))


def fitfun2(x, G, Tn):
    T = 0.012
    return fitfunc(x, G, Tn, T)*1e9

xdata = range(d1.pt)
ydata = PD1[50]*1e9
# initguess = [G, Tn, T]
initguess = [G, Tn]

popt, pcov = curve_fit(fitfun2, xdata, ydata, p0=initguess)
G2, Tn2 = popt

y2 = fitfun2(xdata, G2, Tn2)
plt.figure(2)
plt.plot(d1.lin, ydata)
plt.plot(d1.lin, y2)

# header1 = make_header(d.n1, d.n2, d.n3, meas_data=('Pow [W]'))
# savemtx('mtx_out//' + filein + '.mtx', MAT1, header=header1)
