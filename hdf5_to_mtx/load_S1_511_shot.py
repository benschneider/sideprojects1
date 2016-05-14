import numpy as np
from parsers import load_hdf5, dim
from parsers import savemtx, make_header
# import matplotlib.pyplot as plt
# from changeaxis import interp_y
# from scipy.constants import Boltzmann as Kb
# from scipy.constants import h  , e, pi

filein = "S1_511_shot_100mV_4924_5217MHz"
folder = "hdf5s//09//Data_0915//"
d = load_hdf5(folder+filein+'.hdf5')

# # meas specific to change mag field to flux
# # simply comment this paragraph out
# xoff = 140.5e-3  # 139.3e-3
# x1flux = 479.6e-3
# d.n2.lin = (d.n2.lin-xoff)/x1flux + 0.5
# d.n2.start = d.n2.lin[0]
# d.n2.stop = d.n2.lin[-1]
# d.n2.name = 'Flux/Flux0'

d.n2 = [dim(name=d.stepInst[0],
            start=sPar[3],
            stop=sPar[4],
            pt=sPar[8],
            scale=1)
        for sPar in d.stepItems[0]]


def search(chanList, searchString):
    for i, k in enumerate(chanList):
        if searchString in k:
            return i, k
    return None


def get_MP(d, chnum):
    compx = 1j*d.data[:, chnum+1, :]
    compx += d.data[:, chnum, :]
    return np.abs(compx), np.angle(compx)

MAT1 = np.zeros([7, d.shape[0], d.shape[1]])
MAT1[0] = d.data[:, 1, :]
MAT1[1] = d.data[:, 2, :]
MAT1[2], MAT1[3] = get_MP(d, 7)
MAT1[4], MAT1[5] = get_MP(d, 9)
MAT1[-1] = d.data[:, -1, :]

M2 = np.zeros((7, d.n2[0].pt, d.n3.pt))
M3 = np.zeros((7, d.n2[1].pt, d.n3.pt))
M3 = MAT1[:, :d.n2[0].pt, :]
M2 = MAT1[:, d.n2[0].pt-1:, :]
header2 = make_header(d.n3, d.n2[1], d.n1, meas_data=('a.u.'))
header1 = make_header(d.n3, d.n2[0], d.n1, meas_data=('a.u.'))
savemtx('mtx_out//' + filein + '.mtx', M3, header=header1)
savemtx('mtx_out//' + filein + '2' + '.mtx', M2, header=header2)
