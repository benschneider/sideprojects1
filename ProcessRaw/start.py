import numpy as np
from matplotlib import pyplot as plt
from covfunc import getCovMatrix
from scipy import signal

numtaps = 100
f = 0.2
fir1 = signal.firwin(numtaps, f)

plt.ion()

I1 = np.random.random(int(1e3))  # dummy test data
Q1 = np.random.random(int(1e3))
I2 = np.random.random(int(1e3))
Q2 = np.random.random(int(1e3))

I1new = signal.filtfilt(fir1, 1, I1, padlen=10)


plt.figure()
plt.plot(fir1)
plt.figure()
plt.plot(I1)
plt.figure()
plt.plot(I1new)

lags = 20
matrix = getCovMatrix(I1, Q1, I2, Q2)  # calculate covariance matrix

# plt.figure()
# plt.plot(matrix[0])
# plt.figure()
# plt.plot(matrix[6])
