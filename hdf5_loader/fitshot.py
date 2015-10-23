import numpy as np
from parsers import savemtx, loadmtx
from scipy.optimize import curve_fit
from scipy.constants import Boltzmann as Kb
from scipy.constants import h, e  # , pi
import matplotlib.pyplot as plt

#Start values
Z0 = 50.0
Zopt = 50.0
T = 0.012
R = 75.0
Tn = 3.1
B = 50e6
f = 5800*1e6
G = 59627515

folder = 'mtx_out//'
filein = 'S1_631_SN_G100_BPF4'
M1, header1 = loadmtx('mtx_out//' + filein + '.mtx')

IVs = M1[8]
PD1 = M1[0]
PD2 = M1[1]


def Rm(I):
    return 75.0  # temp solution for now
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
