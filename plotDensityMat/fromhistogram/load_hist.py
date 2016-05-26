import numpy as np
import tables
import matplotlib.pyplot as plt

from parsers import storehdf5 as sh5
from parsers import savemtx

file1 = 'datafold//1128ON//Hist2d.hdf5'
file2 = 'datafold//1128OFF//Hist2d.hdf5'

# fname = 'Hist2d.hdf5'
dh1 = sh5(file1)
dh1.open_f(mode='r')
Data1 = dh1.h5.root
mat1 = Data1.I1I2_2[:]
dh1.close()

dh2 = sh5(file2)
dh2.open_f(mode='r')
Data2 = dh2.h5.root
mat2 = Data2.I1I2_2[:]
dh2.close()

# map3d = Data.I1I2_2[0, 11, :]
# idxp = 2  # this index definex which map is being loaded
# xmin = Data.XminXmaxXNum[idxp][0]
# xmax = Data.XminXmaxXNum[idxp][1]
# xnum = Data.XminXmaxXNum[idxp][2] - 1
# ymin = Data.YminYmaxYNum[idxp][0]
# ymax = Data.YminYmaxYNum[idxp][1]
# ynum = Data.YminYmaxYNum[idxp][2] - 1
# xvec = np.linspace(xmin, xmax, xnum)
# yvec = np.linspace(ymin, ymax, ynum)
#
# # def remap(map2d):
# #     for ii in range(len(map2d[:, 0])):
# #         for jj, value in enumerate(map2d[ii, :]):
# #             return xvec, yvec
#
# fileout = 'out1.mtx'
# header = ('Units,ufo,d1,' + str(xmin) + ',' + str(xmax) +
#           ',d2,' + str(ymin) + ',' + str(ymax) +
#           ',Power,0,1')
# savemtx(fileout, map3d, header)
