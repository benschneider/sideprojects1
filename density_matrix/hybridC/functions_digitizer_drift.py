'''
Functions to track and determine the digitizer drifts
This is done by analysing the digitizer signal at the two frequencies for the case where the pump was switched off.
Since this state is interleaved with the measured data points, changes i.e. LVL corrections/slow drift by the digitizer
can be captured and compensated for.
(This has a similar effect as using a pulsed measurement to ensure relative gain calibration)
'''
import numpy as np
import PyGnuplot as gp

def generate_drift_map(dispData, dicData):
    res = dicData['res']
    off12IQ = dicData['off12'].h5.root.D12raw  # Ia1, Qa1, Ib2, Qb2
    off21IQ = dicData['off21'].h5.root.D12raw  # Ib1, Qb1, Ia2, Qa2
    power1a = np.zeros(off12IQ.shape[0])
    power2a = np.zeros(off12IQ.shape[0])
    power1b = np.zeros(off12IQ.shape[0])
    power2b = np.zeros(off12IQ.shape[0])
    for i in range(off12IQ.shape[0]):
        # This might take some time as it loads a couple of GB of data
        # power1a = I1a**2.0 + Q1a**2.0  # Received Power by digitizer 'a' at frequency 1
        # these are voltage-square averaged Powers
        power1a[i] = np.mean((off12IQ[i][0])**2 + (off12IQ[i][1])**2)  # Ia**2 + Qa**2 @ f1
        power2a[i] = np.mean((off21IQ[i][0])**2 + (off21IQ[i][1])**2)  # Ia**2 + Qa**2 @ f2
        power1b[i] = np.mean((off21IQ[i][2])**2 + (off21IQ[i][3])**2)  # Ib**2 + Qb**2 @ f1
        power2b[i] = np.mean((off12IQ[i][2])**2 + (off12IQ[i][3])**2)  # Ib**2 + Qb**2 @ f2
    res.cpow1a = power1a / np.mean(power1a)
    res.cpow1b = power1b / np.mean(power1b)
    res.cpow2a = power2a / np.mean(power2a)
    res.cpow2b = power2b / np.mean(power2b)
