import numpy as np
from parsers import load_hdf5, dim
from parsers import savemtx, make_header
# import matplotlib.pyplot as plt
from changeaxis import interp_y
from scipy.constants import Boltzmann as Kb
from scipy.constants import h, e, pi

# filein = "S1_511_shot_100mV_4924_5217MHz"
# filein = "S1_514_S11_4924_5217MHz"
# filein = 'S1_478_DCE_MAP_large_coarse'
# filein = 'S1_514_DCE_MAP_high_pow'
# folder = "hdf5s//09//Data_0915//"
# filein = 'S1_515_DCE_MAP_low_pow'
# folder = "hdf5s//09//Data_0916//"
# filein = 'S1_515_DCE_MAP_700mV_pow'
# filein = 'S1_514_DCE_MAP_low_pow'
folder = "hdf5s//09//Data_0912//"
filein = 'S1_482_shot_5090_5019MHz'
d = load_hdf5(folder+filein+'.hdf5')


def get_MP(d, chnum):
    compx = 1j*d.data[:, chnum+1, :]
    compx += d.data[:, chnum, :]
    phase = np.unwrap(zip(*np.angle(compx)))
    return np.abs(compx), zip(*phase)

'''
MAT1 = np.zeros([8, d.shape[0], d.shape[1]])
MAT1[0] = d.data[:, 2, :]
MAT1[1] = d.data[:, 3, :]
MAT1[2], MAT1[3] = get_MP(d, 8)
MAT1[4], MAT1[5] = get_MP(d, 10)
MAT1[6] = d.data[:, 12, :]
MAT1[7] = d.data[:, 13, :]
'''
'''
# scale data to photon number
f1 = 4.924e9
f2 = 5.217e9
# B = 1.37e6
# B = 50e3
B = 5e6
G1 = 2.52057e+07
G2 = 2.16209e+07
MAT1[0] = MAT1[0]/(h*f1*B*G1)
MAT1[1] = MAT1[1]/(h*f2*B*G2)

# meas specific to change mag field to flux
# simply comment this paragraph out
n = 1
d.n3 = [dim(name=d.stepInst[n],
            start=sPar[3],
            stop=sPar[4],
            pt=sPar[8],
            scale=1)
        for sPar in d.stepItems[n]]
d.n3 = d.n3[0]

xoff = 140.5e-3  # 139.3e-3
x1flux = 479.6e-3
d.n3.lin = (d.n3.lin-xoff)/x1flux + 0.5
d.n3.start = d.n3.lin[0]
d.n3.stop = d.n3.lin[-1]
d.n3.name = 'Flux/Flux0'

header1 = make_header(d.n3, d.n2, d.n1, meas_data=('Photons [#]'))
savemtx('mtx_out//' + filein + '.mtx', MAT1, header=header1)

factor = 10
y = (d.n2.lin*d.n2.lin/50.0)  # position of the data
MAT2 = np.zeros([2, d.shape[0]*factor, d.shape[1]])
MAT2[0] =interp_y(y, MAT1[0], factor=factor)
MAT2[1] =interp_y(y, MAT1[1], factor=factor)
y2 = np.linspace(y[0], y[-1], len(y)*factor)
d.dim_y2 = d.n2
d.dim_y2.start = y2[0]
d.dim_y2.stop = y2[-1]
d.dim_y2.pt = len(y2)
d.dim_y2.lin = y2
d.dim_y2.name = 'Pump power (W)'

header2 = make_header(d.n1, d.dim_y2, d.n3, meas_data='Photons [#]')
savemtx('mtx_out//' + filein + 'W' + '.mtx', MAT2, header=header2)

# this is used if a forward and backward sweep was used...
n = 1
d.n3 = [dim(name=d.stepInst[n],
            start=sPar[3],
            stop=sPar[4],
            pt=sPar[8],
            scale=1)
        for sPar in d.stepItems[n]]

d.n2[0].lin =  (d.n2[0].lin-xoff)/x1flux + 0.5
d.n2[0].start = d.n2[0].lin[0]
d.n2[0].stop =  d.n2[0].lin[-1]
d.n2[0].name = 'Flux/Flux0'
d.n2[1].lin =  (d.n2[1].lin-xoff)/x1flux + 0.5
d.n2[1].start = d.n2[1].lin[0]
d.n2[1].stop =  d.n2[1].lin[-1]
d.n2[1].name = 'Flux/Flux0'

M2 = np.zeros((MAT1.shape[0], d.n2[0].pt, d.n3.pt))
M3 = np.zeros((MAT1.shape[0], d.n2[1].pt, d.n3.pt))
M3 = MAT1[:, :d.n2[0].pt, :]
M2 = MAT1[:, d.n2[0].pt-1:, :]

header2 = make_header(d.n3, d.n2[0], d.n1, meas_data=('a.u.'))
header1 = make_header(d.n3, d.n2[1], d.n1, meas_data=('a.u.'))
savemtx('mtx_out//' + filein + '.mtx', M3, header=header1)
savemtx('mtx_out//' + filein + '2' + '.mtx', M2, header=header2)
'''
