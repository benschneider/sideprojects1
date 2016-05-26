import numpy as np
import matplotlib.pyplot as plt
from parsers import storehdf5, loadmtx, savemtx

plt.ion()

file1 = 'RawON.hdf5'
folder1 = 'data3//1150_ON//'
on = storehdf5(folder1 + file1)
on.open_f(mode='r')

file2 = 'RawOFF.hdf5'
folder2 = 'data3//1150_OFF//'
off = storehdf5(folder2 + file2)
off.open_f(mode='r')

d1 = on.h5.root
d2 = off.h5.root


def makehist2d(select):
    on.I1 = d1.D12raw[select][0]
    on.Q1 = d1.D12raw[select][1]
    on.I2 = d1.D12raw[select][2]
    on.Q2 = d1.D12raw[select][3]
    off.I1 = d2.D12raw[select][0]
    off.Q1 = d2.D12raw[select][1]
    off.I2 = d2.D12raw[select][2]
    off.Q2 = d2.D12raw[select][3]
    on.IImap, on.xII, on.yII = np.histogram2d(on.I1, on.I2, [on.xII, on.yII])
    on.QQmap, on.xQQ, on.yQQ = np.histogram2d(on.Q1, on.Q2, [on.xQQ, on.yQQ])
    on.IQmap, on.xIQ, on.yIQ = np.histogram2d(on.I1, on.Q2, [on.xIQ, on.yIQ])
    on.QImap, on.xQI, on.yQI = np.histogram2d(on.Q1, on.I2, [on.xQI, on.yQI])
    off.IImap, off.xII, off.yII = np.histogram2d(off.I1, off.I2, [on.xII, on.yII])
    off.QQmap, off.xQQ, off.yQQ = np.histogram2d(off.Q1, off.Q2, [on.xQQ, on.yQQ])
    off.IQmap, off.xIQ, off.yIQ = np.histogram2d(off.I1, off.Q2, [on.xIQ, on.yIQ])
    off.QImap, off.xQI, off.yQI = np.histogram2d(off.Q1, off.I2, [on.xQI, on.yQI])
    on.IIdmap = on.IImap - off.IImap
    on.QQdmap = on.QQmap - off.QQmap
    on.IQdmap = on.IQmap - off.IQmap
    on.QIdmap = on.QImap - off.QImap


def makeheader(select):
    on.I1 = d1.D12raw[select][0]
    on.Q1 = d1.D12raw[select][1]
    on.I2 = d1.D12raw[select][2]
    on.Q2 = d1.D12raw[select][3]
    off.I1 = d2.D12raw[select][0]
    off.Q1 = d2.D12raw[select][1]
    off.I2 = d2.D12raw[select][2]
    off.Q2 = d2.D12raw[select][3]
    on.IImap, on.xII, on.yII = np.histogram2d(on.I1, on.I2, mapdim)
    on.QQmap, on.xQQ, on.yQQ = np.histogram2d(on.Q1, on.Q2, mapdim)
    on.IQmap, on.xIQ, on.yIQ = np.histogram2d(on.I1, on.Q2, mapdim)
    on.QImap, on.xQI, on.yQI = np.histogram2d(on.Q1, on.I2, mapdim)
    on.headerII = ('Units,ufo,I1,' + str(on.xII[0]) + ',' + str(on.xII[-2]) +
                   ',I2,' + str(on.yII[0]) + ',' + str(on.yII[-2]) + ',DPow,2.03,0.03')
    on.headerQQ = ('Units,ufo,Q1,' + str(on.xQQ[0]) + ',' + str(on.xQQ[-2]) +
                   ',Q2,' + str(on.yQQ[0]) + ',' + str(on.yQQ[-2]) + ',DPow,2.03,0.03')
    on.headerIQ = ('Units,ufo,I1,' + str(on.xIQ[0]) + ',' + str(on.xIQ[-2]) +
                   ',Q2,' + str(on.yIQ[0]) + ',' + str(on.yIQ[-2]) + ',DPow,2.03,0.03')
    on.headerQI = ('Units,ufo,Q1,' + str(on.xQI[0]) + ',' + str(on.xQI[-2]) +
                   ',I2,' + str(on.yQI[0]) + ',' + str(on.yQI[-2]) + ',DPow,2.03,0.03')


def ploth2():
    plt.figure(1)
    plt.imshow(on.IIdmap, interpolation='nearest', origin='low',
               extent=[on.xII[0], on.xII[-1], on.yII[0], on.yII[-1]])
    plt.title('I1I2')
    plt.figure(2)
    plt.imshow(on.QQdmap, interpolation='nearest', origin='low',
               extent=[on.xQQ[0], on.xQQ[-1], on.yQQ[0], on.yQQ[-1]])
    plt.title('Q1Q2')
    plt.figure(3)
    plt.imshow(on.IQdmap, interpolation='nearest', origin='low',
               extent=[on.xIQ[0], on.xIQ[-1], on.yIQ[0], on.yIQ[-1]])
    plt.title('I1Q2')
    plt.figure(4)
    plt.imshow(on.QIdmap, interpolation='nearest', origin='low',
               extent=[on.xQI[0], on.xQI[-1], on.yQI[0], on.yQI[-1]])
    plt.title('Q1I2')


mapdim = [50, 50]
leng = 201
IIm = np.zeros([leng, mapdim[0], mapdim[1]])
QQm = np.zeros([leng, mapdim[0], mapdim[1]])
IQm = np.zeros([leng, mapdim[0], mapdim[1]])
QIm = np.zeros([leng, mapdim[0], mapdim[1]])

makeheader(0)
for i in range(leng):
    makehist2d(i)
    IIm[i] = on.IIdmap
    QQm[i] = on.QQdmap
    IQm[i] = on.IQdmap
    QIm[i] = on.QIdmap

savemtx('IIm.mtx', IIm, on.headerII)
savemtx('QQm.mtx', QQm, on.headerQQ)
savemtx('IQm.mtx', IQm, on.headerIQ)
savemtx('QIm.mtx', QIm, on.headerQI)
# on.close()
# off.close()
