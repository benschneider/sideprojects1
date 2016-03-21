import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.constants import h, e
from scipy.constants import Boltzmann as k
import Gnuplot as gp


N = 20000
T = 1e-11
Tx = np.linspace(0, N*T, N)
s1 = 1  # Test signal stength
s2 = 1  # Carrier signal strength
s0 = 1

f1 = 5e9  # Test signal freq
f2 = 5.2e9  # noise signal
f0 = f1-187e6  # Mixing frequency
phi = 0

sig1 = s1*np.sin(2*pi*f1*Tx+phi)
sig2 = s2*np.sin(2*pi*f2*Tx)

sig0a = s0*np.sin(2*pi*f0*Tx)
sig0b = s0*np.cos(2*pi*f0*Tx)

Isig = sig1*sig0a
Qsig = sig1*sig0b

plt.figure()
plt.plot(Tx, sig1)
plt.figure()
plt.plot(Tx, sig2)
plt.figure()
plt.plot(Tx, Isig)
plt.plot(Tx, Qsig)
# Func. 
# mixed2 = mixed + carrier + noisesig
Fourier = np.fft.fft(Isig)
Freq = np.fft.fftfreq(N, d=T)
plt.figure()
plt.plot(Freq, np.abs(Fourier)/(N-1))


# Filter above 200MHz
LP = int(1/(500e6*T))
LPfft = Fourier[N/2-LP:N/2+LP]
LPfreq = Freq[N/2-LP:N/2+LP]
plt.figure()
plt.plot(LPTx, np.abs(LPfft))

iFourier = np.fft.ifft(LPfft)
plt.figure()
plt.plot(LPTx, abs(iFourier))


resultmat = np.zeros([N, 2])
resultmat[:, 0] = Freq
resultmat[:, 1] = np.abs(Fourier)/(N-1)
#resultmat[:, 2] = np.abs(iFourier)/(N-1)
np.savetxt('test.dat', resultmat, delimiter='\t')
# g1 = gp.Gnuplot(persist=1, debug=1)
# g1("plot 'test.dat' u 1:2 w l")
# g1("unset key")
# g1("replot")

#z = 1000
#resultmat = np.zeros([z, 2])
#resultmat[:, 0] = np.linspace(-10e-6, 10e-6, z)
#for ii, i in enumerate(resultmat[:, 0]):
#    resultmat[ii, 1] = func(i)