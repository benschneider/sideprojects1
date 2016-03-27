import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos
from scipy.constants import h, e
from scipy.constants import Boltzmann as k
import Gnuplot as gp


N = int(1e4)
T = 1e-6  # step size
t = np.linspace(0, N*T, N)  # time
fstep = 1.0/(N*T)

s1 = 1.0
#s2 = 0.0

f1 = 1000
Lo = 187
f2 = f1 + 2*Lo
# phi = 10.0


def digi(signal, f1):
    I = signal*sin(2.0*pi*(f1+Lo)*t)
    Q = signal*sin(2.0*pi*(f1+Lo)*t+pi/2.0)
    return I, Q


sig1 = s1*sin(2*pi*f1*t+phi)
sig2 = s2*sin(2*pi*f2*t)
signal = sig1 + sig2

Isig, Qsig = digi(signal, f1)
Isig1, Qsig1 = digi(sig1, f1)

fftI = np.fft.fftshift(np.fft.fft(Isig))
fftQ = np.fft.fftshift(np.fft.fft(Isig))
freq = np.fft.fftshift(np.fft.fftfreq(N, d=T))

fftI1 = np.fft.fftshift(np.fft.fft(Isig1))
fftQ1 = np.fft.fftshift(np.fft.fft(Isig1))


def LPfft(sinput, lpfreq, T):
    Nlp = int(lpfreq*len(sinput)*T)
    # fftlp = np.fft.fftshift(np.fft.fft(Isig))
    # fftlp[0:(N/2-Nlp)] = 0
    # fftlp[(N/2+Nlp):-1] = 0
    fftlp = np.fft.fft(sinput)  # 0,1,2,3,-3,-2,-1
    fftlp[Nlp+1:-Nlp] = 0.0
    return fftlp


def killsideband(Idata, killup=True):
    smid = int(len(sinput)/2) + 1
    if killup is True:
        sinput[smid:-1] = 0.0
    else:
        sinput[0:smid] = 0.0
    return sinput*2.0


plt.close('all')
plt.ion()
# plt.figure(1)
# plt.plot(t, sig1)
# plt.title('Sig1')
# plt.figure(2)
# plt.plot(t, sig2)
# plt.title('Sig2')
# plt.figure(3)
# plt.plot(t, Isig)
# plt.title('Isig')
# plt.figure(4)
# plt.plot(t, Qsig)
# plt.title('Qsig')
plt.figure(5)
plt.plot(freq, np.abs(fftI-fftI1)/(N-1))
plt.plot(freq, np.abs(fftI1)/(N-1))
plt.title('realfftI')
plt.figure(6)
plt.plot(freq, np.angle(fftI-fftI1)/(N-1))
plt.plot(freq, np.angle(fftI1)/(N-1))
plt.title('imagfftI')

lpI = LPfft(Isig, T, 4000)
lpQ = LPfft(Qsig, T, 4000)
# lpI = killsideband(lpI, killup=True)
# lpQ = killsideband(lpQ, killup=False)
plt.figure(7)
plt.plot(freq, np.fft.fftshift(np.abs(lpI)/(N-1)))
plt.plot(freq, np.fft.fftshift(np.angle(lpI)))
plt.title('lpfftI')
plt.figure(8)
plt.plot(t, np.fft.ifft(lpI))
plt.plot(t, np.fft.ifft(lpQ))
plt.show()


# resultmat = np.zeros([N, 2])
# resultmat[:, 0] = Freq
# resultmat[:, 1] = np.abs(Fourier)/(N-1)
# resultmat[:, 2] = np.abs(iFourier)/(N-1)
# np.savetxt('test.dat', resultmat, delimiter='\t')
# g1 = gp.Gnuplot(persist=1, debug=1)
# g1("plot 'test.dat' u 1:2 w l")
# g1("unset key")
# g1("replot")

# z = 1000
# resultmat = np.zeros([z, 2])
# resultmat[:, 0] = np.linspace(-10e-6, 10e-6, z)
# for ii, i in enumerate(resultmat[:, 0]):
#    resultmat[ii, 1] = func(i)
