from parsers import storehdf5, loadmtx, savemtx
import numpy as np
import matplotlib.pyplot as plt

file1 = 'RawON.hdf5'
folder1 = 'data3//1150_ON//'
file2 = 'RawOFF.hdf5'
folder2 = 'data3//1150_OFF//'

fd1 = storehdf5(folder1+file1)
fd2 = storehdf5(folder2+file2)

fd1.open_f(mode='r')
fd2.open_f(mode='r')

d1 = fd1.h5.root
d2 = fd2.h5.root

'''
on_file = 'Hist2dON.hdf5'
off_file = 'Hist2dOFF.hdf5'
on = storehdf5(on_file)
off = storehdf5(off_file)
on.open_f(mode='r')
off.open_f(mode='r')

on_d = on.h5.root
off_d = off.h5.root

ii = 0
jj = 0
kk = 0
on_I1I2 = on_d.I1I2_2[ii, jj, kk]
on_x = on_d.XminXmaxXNum[ii, jj, kk]
on_y = on_d.YminYmaxYNum[ii, jj, kk]

off_I1I2 = off_d.I1I2_2[ii, jj, kk]
off_x = off_d.XminXmaxXNum[ii, jj, kk]
off_y = off_d.YminYmaxYNum[ii, jj, kk]

on_xvec = np.linspace(on_x[0], on_x[1], on_x[2])
on_yvec = np.linspace(on_y[0], on_y[1], on_y[2])
off_xvec = np.linspace(off_x[0], off_x[1], off_x[2])
off_yvec = np.linspace(off_y[0], off_y[1], off_y[2])

xmin = np.min([on_xvec, off_xvec])
xmax = np.max([on_xvec, off_xvec])
ymin = np.min([on_yvec, off_yvec])
ymax = np.max([on_yvec, off_yvec])

xvec = np.linspace(xmin, xmax, on_x[2])  # new binning vector
yvec = np.linspace(ymin, ymax, on_y[2])

mapshape = on_I1I2.shape
nmap = np.zeros(mapshape)  # the new shape of the map
for ii in range(mapshape[0]):
    for jj in range(mapshape[1]):
        x0 = on_xvec[ii]
        y0 = on_yvec[jj]
        x1 = off_xvec[ii]
        y1 = off_yvec[jj]
'''
