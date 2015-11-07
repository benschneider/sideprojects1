import numpy as np
from parsers import load_hdf5  # , dim
from parsers import savemtx, make_header
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt
# from changeaxis import interp_y
from scipy.constants import Boltzmann as Kb
from scipy.constants import h  # , e , pi

# filein = "S1_511_shot_100mV_4924_5217MHz"
# filein = "S1_514_S11_4924_5217MHz"
# filein = 'S1_478_DCE_MAP_large_coarse'
# filein = 'S1_514_DCE_MAP_high_pow'
# folder = "hdf5s//09//Data_0915//"
# filein = 'S1_515_DCE_MAP_low_pow'
# folder = "hdf5s//09//Data_0916//"
# filein = 'S1_515_DCE_MAP_700mV_pow'
# filein = 'S1_514_DCE_MAP_low_pow'
# folder = "hdf5s//09//Data_0912//"
# filein = 'S1_482_shot_5090_5019MHz'
# folder = "hdf5s//09//Data_0921//"
# filein = 'S1_520_shot_BPF3'
# folder = "hdf5s//10//Data_1022//"
# filein = 'S1_631_SN_G100_BPF4'
# folder = "hdf5s//10//Data_1030//"
# filein = 'S1_655_SN_4p1_4p5_BPF7'
# folder = "hdf5s//10//Data_1028//"
# filein = 'S1_649_DCE_4p1_4p5_BPF7'

folder = 'hdf5s/11/Data_1104//'
filein = 'S1_804_DCE_4p1_4p5_BPF7'
d = load_hdf5(folder+filein+'.hdf5')


def get_MP(d, chnum):
    '''
    This function is used to obtain the magnitude and phase from
    a complex data set.
    It assumes that the next channel is part of the complex number.
    real + i* imaginary
    '''
    compx = 1j*d.data[:, chnum+1, :]
    compx += d.data[:, chnum, :]
    phase = np.unwrap(zip(*np.angle(compx)))
    return np.abs(compx), zip(*phase)

'''
# For Shotnoise Data
MAT1 = np.zeros([9, d.shape[0], d.shape[1]])
MAT1[0] = d.data[:, 9, :]
MAT1[1] = d.data[:, 10, :]
MAT1[2], MAT1[3] = get_MP(d, 2)
MAT1[4], MAT1[5] = get_MP(d, 4)
MAT1[6] = d.data[:, 11, :]
MAT1[7] = d.data[:, 12, :]
MAT1[8] = d.data[:, 13, :]
'''

MAT1 = np.zeros([11, d.shape[0], d.shape[1]])
MAT1[0] = d.data[:, 10, :]
MAT1[1] = d.data[:, 11, :]
MAT1[2], MAT1[3] = get_MP(d, 2)
MAT1[4], MAT1[5] = get_MP(d, 4)
MAT1[6], MAT1[7] = get_MP(d, 6)
MAT1[8], MAT1[9] = get_MP(d, 8)
MAT1[10] = d.data[:, 14, :]

# scale data to photon number
f1 = 4.1e9
f2 = 4.5e9
B = 10e6
G1 = 32910363
Tn1 = 3.875
G2 = 44265492
Tn2 = 3.028
fac1 = (h*f1*B*G1)
fac2 = (h*f2*B*G2)
Namp1 = Tn1*Kb/(h*f1)
Namp2 = Tn2*Kb/(h*f2)

MAT1[0] = MAT1[0]/fac1 - Namp1
MAT1[1] = MAT1[1]/fac2 - Namp2

xoff = 140.5e-3  # 139.3e-3
x1flux = 479.6e-3
d.n1.lin = (d.n1.lin-xoff)/x1flux + 0.5
d.n1.start = d.n1.lin[0]
d.n1.stop = d.n1.lin[-1]
d.n1.name = 'Flux/Flux0'


# scale voltage axis to uV
DCGain = 1000.0
MAT1[10] = MAT1[10]*(1e6/DCGain)

# scale Power axis to units of flux change
dpow = 0.25-0.025
dflux = 0.654295-0.5
powScale = dflux/dpow
d.n2.start = d.n2.start*powScale
d.n2.stop = d.n2.stop*powScale
d.n2.name = 'Pump [Phi/Phi0]'

'''
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
'''

'''
# (x-140.5e-3)/479.6e-3+ 0.5
# header1 = make_header(d.n3, d.n2, d.n1, meas_data=('Photons [#]'))
header1 = make_header(d.n3, d.n2, d.n1, meas_data=('Pow [W]'))
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
'''
'''
# this is used if a forward and backward sweep was used...
n = 0  # channel number of that instrument with multiple sweeps
d.n2 = [dim(name=d.stepInst[n],
            start=sPar[3],
            stop=sPar[4],
            pt=sPar[8],
            scale=1)
        for sPar in d.stepItems[n]]
'''
'''
d.n2[0].lin =  (d.n2[0].lin-xoff)/x1flux + 0.5
d.n2[0].start = d.n2[0].lin[0]
d.n2[0].stop =  d.n2[0].lin[-1]
d.n2[0].name = 'Flux/Flux0'
d.n2[1].lin =  (d.n2[1].lin-xoff)/x1flux + 0.5
d.n2[1].start = d.n2[1].lin[0]
d.n2[1].stop =  d.n2[1].lin[-1]
d.n2[1].name = 'Flux/Flux0'
'''
'''
M2 = np.zeros((MAT1.shape[0], d.n2[0].pt, d.n3.pt))
M4 = np.zeros((MAT1.shape[0], d.n2[1].pt, d.n3.pt))
M2 = MAT1[:, :d.n2[0].pt, :]
M3 = MAT1[:, d.n2[0].pt-1:, :]

header1 = make_header(d.n3, d.n2[0], d.n1, meas_data=('a.u.'))
savemtx('mtx_out//' + filein + '.mtx', M2, header=header1)
header2 = make_header(d.n3, d.n2[1], d.n1, meas_data=('a.u.'))
savemtx('mtx_out//' + filein + '2' + '.mtx', M3, header=header2)
'''

header1 = make_header(d.n1, d.n2, d.n3, meas_data=('Photon Flux'))
# header1 = make_header(d.n1, d.n2, d.n3, meas_data=('Pow [W]'))
savemtx('mtx_out//' + filein + '.mtx', MAT1, header=header1)
