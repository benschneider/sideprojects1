import numpy as np
from parsers import storehdf5, loadmtx, savemtx
from scipy.constants import h  # , e, c , pi
import scipy.signal as signal
from scipy.signal.signaltools import _next_regular
from numpy.fft import rfftn, irfftn


def load_dataset(dispData, dicData, ext='hdf5_on'):
    '''Reads from the dictionary which data set was selected'''
    dicData[ext] = storehdf5(str(dispData[ext]))
    dicData[ext].open_f(mode='r')


def prep_data(dispData, dicData):
    on = dicData['hdf5_on']
    off = dicData['hdf5_off']
    f1 = dispData['f1']
    f2 = dispData['f2']
    G1 = dispData['g1']
    G2 = dispData['g2']
    B = dispData['B']
    Fac1 = (h*f1*G1*B)**0.5
    Fac2 = (h*f2*G2*B)**0.5
    return on, off, Fac1, Fac2


def assignRaw(dispData, dicData):
    select = dispData['select']
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    d1 = on.h5.root
    d2 = off.h5.root
    on.I1 = d1.D12raw[select][0]/Fac1  # units in sqrt Photon numbers
    on.Q1 = d1.D12raw[select][1]/Fac1
    on.I2 = d1.D12raw[select][2]/Fac2
    on.Q2 = d1.D12raw[select][3]/Fac2
    off.I1 = d2.D12raw[select][0]/Fac1
    off.Q1 = d2.D12raw[select][1]/Fac1
    off.I2 = d2.D12raw[select][2]/Fac2
    off.Q2 = d2.D12raw[select][3]/Fac2


def makehist2d(dispData, dicData):
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
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


def makeheader(dispData, dicData):
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    mapdim = dispData['mapdim']
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


def getCovMatrix(I1, Q1, I2, Q2, lags=20):
    # calc <I1I2>, <I1Q2>, Q1I2, Q1Q2
    lags = int(lags)
    I1 = np.asarray(I1)
    Q1 = np.asarray(Q1)
    I2 = np.asarray(I2)
    Q2 = np.asarray(Q2)
    CovMat = np.zeros([6, lags*2-1])
    start = len(I1*2-1)-lags
    stop = len(I1*2-1)-1+lags
    sI1 = np.array(I1.shape)
    sQ2 = np.array(Q2.shape)
    shape = sI1 + sQ2 - 1
    HPfilt = (int(sI1/(lags*4)))  # smallest features visible is lamda/4
    fshape = [_next_regular(int(d)) for d in shape]  # padding to optimal size for FFTPACK
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    # Do FFTs and get Cov Matrix
    fftI1 = rfftn(I1, fshape)
    fftQ1 = rfftn(Q1, fshape)
    fftI2 = rfftn(I2, fshape)
    fftQ2 = rfftn(Q2, fshape)
    rfftI1 = rfftn(I1[::-1], fshape)
    rfftQ1 = rfftn(Q1[::-1], fshape)
    rfftI2 = rfftn(I2[::-1], fshape)
    rfftQ2 = rfftn(Q2[::-1], fshape)
    # filter frequencies outside the lags range
    fftI1 = np.concatenate((np.zeros(HPfilt), fftI1[HPfilt:]))
    fftQ1 = np.concatenate((np.zeros(HPfilt), fftQ1[HPfilt:]))
    fftI2 = np.concatenate((np.zeros(HPfilt), fftI2[HPfilt:]))
    fftQ2 = np.concatenate((np.zeros(HPfilt), fftQ2[HPfilt:]))
    # filter frequencies outside the lags range
    rfftI1 = np.concatenate((np.zeros(HPfilt), rfftI1[HPfilt:]))
    rfftQ1 = np.concatenate((np.zeros(HPfilt), rfftQ1[HPfilt:]))
    rfftI2 = np.concatenate((np.zeros(HPfilt), rfftI2[HPfilt:]))
    rfftQ2 = np.concatenate((np.zeros(HPfilt), rfftQ2[HPfilt:]))
    CovMat[0, :] = (irfftn((fftI1*rfftI2))[fslice].copy()[start:stop] / len(fftI1))  # 0: <I1I2>
    CovMat[1, :] = (irfftn((fftQ1*rfftQ2))[fslice].copy()[start:stop] / len(fftI1))  # 1: <Q1Q2>
    CovMat[2, :] = (irfftn((fftI1*rfftQ2))[fslice].copy()[start:stop] / len(fftI1))  # 2: <I1Q2>
    CovMat[3, :] = (irfftn((fftQ1*rfftI2))[fslice].copy()[start:stop] / len(fftI1))  # 3: <Q1I2>
    CovMat[4, :] = (abs(1j*(CovMat[2, :]+CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :])))  # 4: <Squeezing> Magnitude
    CovMat[5, :] = np.angle(1j*(CovMat[2, :]+CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))  # 5: <Squeezing> Angle
    return CovMat


def f1pN2(tArray, d=1):
    lags0 = np.round(len(tArray)/2.0)
    distance = 0
    if np.max(np.abs(tArray[lags0-d:lags0+d+1])) > 2.0*np.var(tArray):
        distance = (np.argmax(np.abs(tArray[lags0-d:lags0+d+1])) - d)*-1
    return distance


def f1pN(tArray, lags0, d=1):
    return (np.argmax(tArray[lags0-d:lags0+d+1]) - d)*-1


def correctPhase(dispData, dicData):
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    lags = dispData['lags']
    CovMat = getCovMatrix(on.I1, on.Q1, on.I2, on.Q2)
    offset = CovMat[5][lags]  # want this phase angle to be zero
    phase = np.angle(1j*on.Q1 + on.I1)  # phase rotation
    new = np.abs(1j*on.Q1 + on.I1)*np.exp(1j*(phase - offset))
    on.I1 = np.real(new)
    on.Q1 = np.imag(new)


def process(dispData, dicData):
    assignRaw(dispData, dicData)
    makeheader(dispData, dicData)
    assignRaw(dispData, dicData)
    correctPhase(dispData, dicData)
    makehist2d(dispData, dicData)  # all data is stored in dicData

# mapdim = [200, 200]  # decide map dimensions
# leng = 200  # how many points to take into
# dy = 10
# assignRaw(0)  # aquire data points
# makeheader(0, mapdim)  # create headers and define histogram axis
#
# IIm = np.zeros([leng, mapdim[0], mapdim[1]])
# QQm = np.zeros([leng, mapdim[0], mapdim[1]])
# IQm = np.zeros([leng, mapdim[0], mapdim[1]])
# QIm = np.zeros([leng, mapdim[0], mapdim[1]])
# # IQ1m = np.zeros([leng, mapdim[0], mapdim[1]])
# # IQ2m = np.zeros([leng, mapdim[0], mapdim[1]])
#
# for i in range(leng):  #this is the main loop
#     for j in range(dy):
#         assignRaw(i+j*201)
#         correctPhase()
#         makehist2d(i+j*201)
#         IIm[i] += on.IIdmap
#         QQm[i] += on.QQdmap
#         IQm[i] += on.IQdmap
#         QIm[i] += on.QIdmap
#         # IQ1m[i] = on.IQ1dmap
#         # IQ2m[i] = on.IQ2dmap
#
# savemtx(fnum+'II.mtx', IIm, on.headerII)
# savemtx(fnum+'QQ.mtx', QQm, on.headerQQ)
# savemtx(fnum+'IQ.mtx', IQm, on.headerIQ)
# savemtx(fnum+'QI.mtx', QIm, on.headerQI)
# # savemtx(fnum+'IQ1m.mtx', IQ1m, on.headerIQ1)
# # savemtx(fnum+'IQ2m.mtx', IQ2m, on.headerIQ2)
# on.close()
# off.close()
#
