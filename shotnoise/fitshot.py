import numpy as np
from parsers import savemtx, loadmtx
from scipy.optimize import curve_fit
from scipy.constants import Boltzmann as Kb
from scipy.constants import h, e  # , pi
import matplotlib.pyplot as plt

# Start values
Z0 = 50.0
Zopt = 50.0
T = 0.012
R = 75.0
Tn = 3.1
B = 50e6
f = 5800*1e6
G = 59627515
RTR = 1009.1  # Resistance at RT in kOhm in front of V output

folder = 'mtx_out//'
filein = 'S1_631_SN_G100_BPF4'
M1, d1, d2, d3, dz = loadmtx('mtx_out//' + filein + '.mtx')

d1.scale = 1000.0/RTR  # scale to µAmp
d1.update_lin()

IVs = M1[8]
IVs = IVs * 1000.0 # scale to µVolts
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
plt.plot(d1.lin, dIVm)


def find_nearest(someArray, value):
    idx = abs(someArray-value).argmin()
    return idx


#Fitting functions:


def Rm(I):
    # pos = find_nearest()
    return dIV[I]
    # return 75.0  # temp solution for now
    # Rm(R) = abs(R)>200.0 ? 100.0 : abs(R)<1.0 ? 5.0 : abs(R)


def E1(I, R, T, f):
    return (e*I*R+h*f)/(2*Kb*T)


def E2(I, R, T, f):
    return (e*I*R-h*f)/(2*Kb*T)


def Si(I, R, f, T):
    return ((2*Kb*T/R)
            * (E1(I, R, T, f)/np.tanh(E1(I, R, T, f))
               + E2(I, R, T, f)/np.tanh(E2(I, R, T, f))))


def func(I, G, Tn, T):
    return (B*G*(Si(I, Rm(I), f, T) * Rm(I)**2 + 4.0*Kb*T*Rm(I)
                 + 4.0*Kb*Tn*Z0*(Rm(I)**2 + Zopt*Zopt)/(Z0*Z0 + Zopt*Zopt))
            * (Z0/((Rm(I)+Z0)*(Rm(I)+Z0))))

xdata = np.linspace(-25e-6, 25e-6, 5001)
y = func(xdata, G, Tn, T)
ydata = y + 1e-7 * np.random.normal(size=len(xdata))


initguess = [G, Tn, T]
popt, pcov = curve_fit(func, xdata, ydata, p0=initguess)
G2, Tn2, T2 = popt
y2 = func(xdata, G2, Tn2, T2)

plt.plot(xdata, ydata)
plt.plot(xdata, y2)

# fit [-19:-5][:] Pout4(x, y)  dFileN u ($1/14.27e-3):($5):($3) via G, Tn
# header1 = make_header(d.n1, d.n2, d.n3, meas_data=('Pow [W]'))
# savemtx('mtx_out//' + filein + '.mtx', MAT1, header=header1)
