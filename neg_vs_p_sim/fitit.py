from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from numpy import pi, cos, sin
import numpy as np


# Simple model for a perfect SQUID at the end of a perfectly matched transmission line

R = 10e3
Ic = 3.4e-6
Cap = 5e-14
flux0 = 2.07e-15    # Tm^2; Flux quanta: flux0 =  h / (2*charging energy)
flux = np.linspace(-0.75, 0.7011, 701) * flux0
offset = -3.5

def fitFunc(flux, R, Ic, Cap, offset):
    flux0 = 2.07e-15
    f0 = 4.1e9
    z0 = 50
    L = flux0 / (Ic * 2.0 * pi * np.abs(cos(pi * flux / flux0)))
    Ysq = (1.0 / R + 1.0 / (1j * 2.0 * pi * f0 * L + 1j * 1e-90) + 1j * 2.0 * pi * f0 * Cap)
    zsq = 1.0 / Ysq
    s11 = (zsq - z0) / (zsq + z0)
    return np.angle(s11)

iguess = [R, Ic, Cap, offset]
temp = fitFunc(flux, R, Ic, Cap, offset)
testdata = temp + 0.25*np.random.normal(size=len(temp))

popt, pcov = curve_fit(fitFunc, flux, testdata, p0=iguess)
print(popt)
print(pcov)

plt.figure(2)
plt.plot(flux/flux0, testdata)
plt.plot(flux/flux0, fitFunc(flux, popt[0], popt[1], popt[2], popt[3]))


