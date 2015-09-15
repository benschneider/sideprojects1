import numpy as np
from parsers import load_hdf5
from parsers import savemtx, make_header
# import matplotlib.pyplot as plt
# from changeaxis import interp_y
# from scipy.constants import Boltzmann as Kb
# from scipy.constants import h  , e, pi

filein = "S1_469_shot_127mV_wide_2"
folder = "hdf5s//09//Data_0906//"

d = load_hdf5(folder+filein+'.hdf5')

# # meas specific to change mag field to flux
# # simply comment this paragraph out
# xoff = 140.5e-3  # 139.3e-3
# x1flux = 479.6e-3
# d.n2.lin = (d.n2.lin-xoff)/x1flux + 0.5
# d.n2.start = d.n2.lin[0]
# d.n2.stop = d.n2.lin[-1]
# d.n2.name = 'Flux/Flux0'

def search(chanList, searchString):
    for i, k in enumerate(chanList):
        if searchString in k:
            return i, k
    return None


# search and collect Real and Imaginary pieces to calc Magnitude and phase...
numOfDigi = 2
MAT1 = np.zeros([(len(d.channel[5:])+2*numOfDigi), d.shape[0], d.shape[1]])
startSearch = 0
for n in range(numOfDigi):
    realPos, InstrumStr = search(d.channel[startSearch:], 'Real')
    if InstrumStr is not None:
        imgPos, InstrumStr2 = search(d.channel[realPos+1:], InstrumStr[0])
        imgPos = imgPos + realPos + 1
    MAT1[n*4] = d.data[:, realPos, :]  # Real data (MagAvg)
    MAT1[n*4+1] = d.data[:, imgPos, :]  # Imag data (MagAvg)
    compx = 1j*MAT1[n*4]
    compx += MAT1[n*4+1]
    MAT1[n*4+2] = np.absolute(compx)
    MAT1[n*4+3] = np.angle(compx)

# store Power channels
MAT1[(numOfDigi)*4] = d.data[:, 5, :]
MAT1[(numOfDigi)*4+1] = d.data[:, 8, :]
MAT1[(numOfDigi)*4+2] = d.data[:, -1, :]  # Store measured Voltage

M2 = np.zeros((11, 1501, 70))
M3 = np.zeros((11, 1501, 70))

M3 = MAT1[:, :1501, :]
M2 = MAT1[:, 1500:, :]
d.n2.pt = (d.n2.pt-1)/2 +1

d.n2.stop = d.n2.stop*-1
d.n2.update_lin()
header1 = make_header(d.n1, d.n2, d.n2, meas_data=('a.u.'))
savemtx('mtx_out//' + filein + '.mtx', M3, header=header1)

#flip endpoints for the backsweep
d.n2.stop = d.n2.stop*-1
d.n2.start = d.n2.start*-1
d.n2.update_lin()
header1 = make_header(d.n1, d.n2, d.n2, meas_data=('a.u.'))
savemtx('mtx_out//' + filein +'2'+ '.mtx', M2, header=header1)

# for jj, filext in enumerate(d.channel[5:]):
#     MAT2[jj+2] = d.data[:, jj+5, :]
#     header1 = make_header(d.n1, d.n2, d.n2, meas_data=filext[1])
#     # savemtx(folder + filein + filext[0] + '.mtx', MAT1[jj+2], header=header1)
