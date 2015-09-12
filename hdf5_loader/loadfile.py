import numpy as np
from parsers import load_hdf5
from parsers import get_hdf5_dataset
from parsers import savemtx, make_header
# import matplotlib.pyplot as plt
from changeaxis import interp_y

filename = "hdf5s//09//Data_0911//S1_475_DCE_MAP_6_fine.hdf5"
d = load_hdf5(filename)
d.c8 = get_hdf5_dataset(d, 8)  # Real data (MagAvg)
d.c9 = get_hdf5_dataset(d, 9)  # Imag data (MagAvg)
d.D1compx = 1j*d.c9.d
d.D1compx += d.c8.d
d.D1mag = np.absolute(d.D1compx)
d.D1phase = np.angle(d.D1compx)

d.c10 = get_hdf5_dataset(d, 10)  # Real data (MagAvg)
d.c11 = get_hdf5_dataset(d, 11)  # Imag data (MagAvg)
d.D2compx = 1j*d.c11.d
d.D2compx += d.c10.d
d.D2mag = np.absolute(d.D2compx)
d.D2phase = np.angle(d.D2compx)

xoff = 139.3e-3
x1flux = 479.6e-3
d.dim_1.lin = (d.dim_1.lin-xoff)/x1flux + 0.5
d.dim_1.start = d.dim_1.lin[0]
d.dim_1.stop = d.dim_1.lin[-1]

MAT1 = np.zeros([6, d.shape[0], d.shape[2]])
MAT1[0] = d.data[:, 3, :]
MAT1[1] = d.data[:, 4, :]
MAT1[2] = d.D1mag
MAT1[3] = d.D1ang
MAT1[4] = d.D1pow
MAT1[5] = d.D1lev

y = (d.dim_2.lin*d.dim_2.lin/50.0)  # position of the data
y2, MAT2 = interp_y(y, d.D1pow)
d.dim_y2 = d.dim_2
d.dim_y2.start = y2[0]
d.dim_y2.stop = y2[-1]
d.dim_y2.pt = len(y2)
d.dim_y2.lin = y2

header1 = make_header(d.dim_1, d.dim_2, d.dim_3, meas_data='(a.u)')
savemtx('output//out1.mtx', MAT1, header=header1)

header2 = make_header(d.dim_1, d.dim_y2, d.dim_3, meas_data='(a.u)')
savemtx('output//out2.mtx', MAT2, header=header2)
