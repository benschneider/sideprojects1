import numpy as np
from parsers import load_hdf5
from parsers import savemtx, make_header
# import matplotlib.pyplot as plt
from changeaxis import interp_y
from scipy.constants import Boltzmann as Kb
from scipy.constants import h, e, pi

filename = "hdf5s//09//Data_0911//S1_475_DCE_MAP_6_fine.hdf5"
d = load_hdf5(filename)
d.c8 = d.data[:, 8, :]  # Real data (MagAvg)
d.c9 = d.data[:, 9, :]  # Imag data (MagAvg)
d.D1compx = 1j*d.c9
d.D1compx += d.c8
d.D1mag = np.absolute(d.D1compx)
d.D1phase = np.angle(d.D1compx)
d.D1Pow = d.data[:, 6, :]

d.c10 = d.data[:, 10, :]   # Real data (MagAvg)
d.c11 = d.data[:, 11, :]  # Imag data (MagAvg)
d.D2compx = 1j*d.c11
d.D2compx += d.c10
d.D2mag = np.absolute(d.D2compx)
d.D2phase = np.angle(d.D2compx)
d.D2Pow = d.data[:, 7, :]

# scale data to photon number
D1AvgP = np.mean(d.D1Pow[-1, :])  # background power
D2AvgP = np.mean(d.D2Pow[-1, :])
f1 = 5.135e9
f2 = 5.145e9
D1Psc = Kb*2.5 / (h*f1*D1AvgP)  # Assume residual noise was 2.5K
D2Psc = Kb*2.5 / (h*f2*D2AvgP)
d.D1Pow = d.D1Pow * D1Psc
d.D2Pow = d.D2Pow * D2Psc

# d.n1 #Yoko
# d.n2 #Pump pow
# d.n3 #nothing c

xoff = 139.3e-3
x1flux = 479.6e-3
d.n1.lin = (d.n1.lin-xoff)/x1flux + 0.5
d.n1.start = d.n1.lin[0]
d.n1.stop = d.n1.lin[-1]
d.n1.name = 'Flux/Flux0'

MAT1 = np.zeros([6, d.shape[0], d.shape[1]])
MAT1[0] = d.D1mag
MAT1[1] = d.D2mag
MAT1[2] = d.D1phase
MAT1[3] = d.D2phase
MAT1[4] = d.D1Pow
MAT1[5] = d.D2Pow


header1 = make_header(d.n1, d.n2, d.n2, meas_data='est. Photon #')
savemtx('output//out1.mtx', MAT1, header=header1)

y = (d.n2.lin*d.n2.lin/50.0)  # position of the data
y2, MAT2 = interp_y(y, d.D1Pow)
d.dim_y2 = d.n2
d.dim_y2.start = y2[0]
d.dim_y2.stop = y2[-1]
d.dim_y2.pt = len(y2)
d.dim_y2.lin = y2
d.dim_y2.name = 'Pump power (W)'

header2 = make_header(d.n1, d.dim_y2, d.n2, meas_data='est. Photon #')
savemtx('output//out2.mtx', MAT2, header=header2)
