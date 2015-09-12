import numpy as np
from scipy.constants import h, e  # , pi
from scipy.constants import Boltzmann as Kb
import matplotlib.pyplot as plt


def get_Si(f, V, T, R):
    ''' to obtain the current noise spectral density ( A^2 / Hz ) '''
    Et = 2*Kb*T
    Ep1 = e*V + h*f
    Ep2 = e*V - h*f
    return (Et/R)*( (Ep1/Et)/(np.tanh(Ep1/Et)) + (Ep2/Et)/np.tanh(Ep2/Et) )


def get_NP(f, V, T, R, Bw, G, Tn):
    Tau2 = np.abs((R-50)/(R+50))**2
    T2 = (1-Tau2)
    Et = 2*Kb*T
    Ep1 = e*V + h*f
    Ep2 = e*V - h*f
    Si = ( (Ep1/Et)/(np.tanh(Ep1/Et)) + (Ep2/Et)/np.tanh(Ep2/Et) )
    return G*Bw*Kb*(Tn + (50/R)*T2*0.5*Si)

x = np.linspace(-40, 40, 1000)*1e-6
y = get_Si(5e9, x, 10e-3, 50)
y2 = get_Si(5e9, x, 10e-3, 74)

plt.figure(1)
plt.plot(x, y)
plt.plot(x, y2)

V = np.linspace(-40, 40, 1000)*1e-6
f = 5e9
T = 10e-3
R = 74.8
Bw = 1e6
G = 10**(8.2)
Tn = 2.5
y3 = get_NP(f, V, T, R, Bw, G, Tn)

plt.figure(2)
plt.plot(V, y3)
plt.show()
