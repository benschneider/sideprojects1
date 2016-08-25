import numpy as np
from parsers import storehdf5, loadmtx, savemtx
from scipy.constants import h  # , e, c , pi
import scipy.signal as signal
from scipy.signal.signaltools import _next_regular
from scipy.ndimage.filters import gaussian_filter1d
from numpy.fft import rfftn, irfftn
import logging


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
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    select = dispData['select']
    LP = dispData['Low Pass']
    d1 = on.h5.root
    d2 = off.h5.root
    on.I1 = lowPass(d1.D12raw[select][0]/Fac1, LP=LP)
    on.Q1 = lowPass(d1.D12raw[select][1]/Fac1, LP=LP)
    on.I2 = lowPass(d1.D12raw[select][2]/Fac2, LP=LP)
    on.Q2 = lowPass(d1.D12raw[select][3]/Fac2, LP=LP)
    off.I1 = lowPass(d2.D12raw[select][0]/Fac1, LP=LP)
    off.Q1 = lowPass(d2.D12raw[select][1]/Fac1, LP=LP)
    off.I2 = lowPass(d2.D12raw[select][2]/Fac2, LP=LP)
    off.Q2 = lowPass(d2.D12raw[select][3]/Fac2, LP=LP)
    dispData['I,Q data length'] = len(on.I1)
    dispData['Current Section i, j, k'] = d1.ijk[select]


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


def getCovMatrix(I1i, Q1i, I2i, Q2i, lags=20, hp=True):
    # calc <I1I2>, <I1Q2>, Q1I2, Q1Q2
    lags = int(lags)
    I1 = np.asarray(I1i)
    Q1 = np.asarray(Q1i)
    I2 = np.asarray(I2i)
    Q2 = np.asarray(Q2i)
    CovMat = np.zeros([6, lags*2+1])
    start = len(I1*2-1)-lags-1
    stop = len(I1*2-1)+lags
    sI1 = np.array(I1.shape)
    sQ2 = np.array(Q2.shape)
    shape = sI1 + sQ2 - 1
    HPfilt = (int(sI1/(lags*2)))  # largest features visible is lamda*4
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
    if hp:
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

    # 0: <I1I2>
    # 1: <Q1Q2>
    # 2: <I1Q2>
    # 3: <Q1I2>
    # 4: <Squeezing> Magnitude
    # 5: <Squeezing> Angle
    CovMat[0, :] = (irfftn((fftI1*rfftI2))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[1, :] = (irfftn((fftQ1*rfftQ2))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[2, :] = (irfftn((fftI1*rfftQ2))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[3, :] = (irfftn((fftQ1*rfftI2))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[4, :] = abs(1j*(CovMat[2, :]+CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))
    CovMat[5, :] = np.angle(1j*(CovMat[2, :]+CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))
    return CovMat


def getCovMat_wrap(dispData, on):
    segment = dispData['Segment Size']
    lags = dispData['lags']
    hp = dispData['FFT-Filter']
    I1 = on.I1
    Q1 = on.Q1
    I2 = on.I2
    Q2 = on.Q2
    num = 0

    if segment:
        num = len(I1)/segment
        modulo = len(I1) % segment  # 4/3 = 1 + residual -> would give an error
        if modulo:
            I1 = I1[:-modulo]  # remove residual data points
            Q1 = Q1[:-modulo]
            I2 = I2[:-modulo]
            Q2 = Q2[:-modulo]

        I1 = np.reshape(I1, [num, segment])
        Q1 = np.reshape(Q1, [num, segment])
        I2 = np.reshape(I2, [num, segment])
        Q2 = np.reshape(Q2, [num, segment])

        CovMat = np.zeros([6, dispData['lags']*2+1])
        for i in range(num):
            CovMat += getCovMatrix(I1[i], Q1[i], I2[i], Q2[i], lags=lags, hp=hp)

        CovMat = CovMat / np.float(num)
        CovMat[4, :] = abs(1j*(CovMat[2, :]+CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))
        CovMat[5, :] = np.angle(1j*(CovMat[2, :]+CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))

    else:
        CovMat = getCovMatrix(I1, Q1, I2, Q2, lags=lags, hp=hp)

    return CovMat


def f1pN2(tArray, lags0, d=1):
    if np.max(np.abs(tArray[lags0-d:lags0+d+1])) > 4.0*np.var(np.abs(tArray)):
        distance = (np.argmax(tArray[lags0-d:lags0+d+1]) - d)*-1
    else:
        distance = 0
        logging.debug('SN ratio too low: Can not find trigger position')
    return distance


def f1pN(tArray, lags0, d=1):
    return (np.argmax(tArray[lags0-d:lags0+d+1]) - d)*-1


def correctPhase(dispData, dicData):
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)

    if dispData['Trigger correction']:
        CovMat = getCovMat_wrap(dispData, on)
        dMag = f1pN2(CovMat[4], dispData['lags'], d=1)
        logging.debug('Trigger correct ' + str(dMag) + 'pt')
        on.I1 = np.roll(on.I1, dMag)  # Correct 1pt trigger jitter
        on.Q1 = np.roll(on.Q1, dMag)

    if dispData['Phase correction']:
        CovMat = getCovMat_wrap(dispData, on)
        phase_index = np.argmax(CovMat[4])
        phase_offset = CovMat[5][phase_index]
        phase = np.angle(1j*on.Q1 + on.I1)  # phase rotation
        new = np.abs(1j*on.Q1 + on.I1)*np.exp(1j*(phase - phase_offset))
        on.I1 = np.real(new)
        on.Q1 = np.imag(new)

    CovMat = getCovMat_wrap(dispData, on)
    getCovMat_wrap(dispData, on)
    on.PSI_mag = CovMat[4]  # record the PSI magnitude value
    on.PSI_phs = CovMat[5]
    on.cII = CovMat[0]
    on.cQQ = CovMat[1]
    on.cIQ = CovMat[2]
    on.cQI = CovMat[3]


def lowPass(data, LP=0):
    if bool(LP):
        data = gaussian_filter1d(data, LP)  # Gausfilter
    return data


def process(dispData, dicData):
    assignRaw(dispData, dicData)
    makeheader(dispData, dicData)
    assignRaw(dispData, dicData)
    correctPhase(dispData, dicData)
    makehist2d(dispData, dicData)
    if bool(dispData['Averages']):
        avg(dispData, dicData)


def avg(dispData, dicData):
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    on.IImap_avg = np.zeros_like(on.IImap)
    on.QQmap_avg = np.zeros_like(on.QQmap)
    on.IQmap_avg = np.zeros_like(on.IQmap)
    on.QImap_avg = np.zeros_like(on.QImap)
    # on.PSI_mag_avg = np.zeros_like(on.PSI_mag)
    # on.PSI_phs_avg = np.zeros_like(on.PSI_phs)
    on.cII_avg = np.zeros_like(on.cII)
    on.cQQ_avg = np.zeros_like(on.cQQ)
    on.cIQ_avg = np.zeros_like(on.cIQ)
    on.cQI_avg = np.zeros_like(on.cQI)
    for i in range(dispData['Averages']):
        logging.debug('Working on trace number '+str(i+1))
        assignRaw(dispData, dicData)
        correctPhase(dispData, dicData)
        makehist2d(dispData, dicData)
        on.IImap_avg += on.IImap
        on.QQmap_avg += on.QQmap
        on.IQmap_avg += on.IQmap
        on.QImap_avg += on.QImap
        on.cII_avg += on.cII
        on.cQQ_avg += on.cQQ
        on.cIQ_avg += on.cIQ
        on.cQI_avg += on.cQI
        # on.PSI_mag_avg += on.PSI_mag
        # on.PSI_phs_avg += on.PSI_phs
        dispData['select'] = dispData['select'] + 201  # go to next data set

    on.IImap = on.IImap_avg/dispData['Averages']
    on.IImap = on.IImap_avg/dispData['Averages']
    on.QQmap = on.QQmap_avg/dispData['Averages']
    on.IQmap = on.IQmap_avg/dispData['Averages']
    on.QImap = on.QImap_avg/dispData['Averages']
    on.cII = on.cII_avg/dispData['Averages']
    on.cQQ = on.cQQ_avg/dispData['Averages']
    on.cIQ = on.cIQ_avg/dispData['Averages']
    on.cQI = on.cQI_avg/dispData['Averages']
    on.PSI_mag = np.abs(1j*(on.cIQ + on.cQI) + (on.cII - on.cQQ))
    on.PSI_phs = np.angle(1j*(on.cIQ + on.cQI) + (on.cII - on.cQQ))
