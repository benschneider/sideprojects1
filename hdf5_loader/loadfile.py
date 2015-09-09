import numpy as np
from parsers import load_hdf5
from parsers import savemtx, make_header

d = load_hdf5('S1_471_DCE_MAP2.hdf5')
d.cD1 = 1j*d.data[:, 6, :]
d.cD1 += d.data[:, 5, :]
d.D1mag = np.absolute(d.cD1)
d.D1ang = np.angle(d.cD1)
d.D1pow = d.data[:, 7, :]
d.D1lev = d.data[:, 8, :]

d.shape = d.data.shape

MAT1 = np.zeros([6, d.shape[0], d.shape[2]])
MAT1[0] = np.imag(d.cD1)
MAT1[1] = np.real(d.cD1)
MAT1[2] = d.D1mag
MAT1[3] = d.D1ang
MAT1[4] = d.D1pow
MAT1[5] = d.D1lev


header1 = make_header(d.dim_1, d.dim_2, d.dim_3, meas_data='(a.u)')
savemtx('output.mtx', MAT1, header=header1)
