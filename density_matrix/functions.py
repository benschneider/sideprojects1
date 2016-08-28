import numpy as np
from parsers import storehdf5, loadmtx, savemtx
from scipy.constants import h  # , e, c , pi
import scipy.signal as signal
from scipy.signal.signaltools import _next_regular
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from numpy.fft import rfftn, irfftn
import logging
import PyGnuplot as gp


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
    d1 = on.h5.root
    LP = dispData['Low Pass']
    d2 = off.h5.root
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    on.I1 = lowPass(d1.D12raw[select][0]/Fac1, LP=LP)
    on.Q1 = lowPass(d1.D12raw[select][1]/Fac1, LP=LP)
    on.I2 = lowPass(d1.D12raw[select][2]/Fac2, LP=LP)
    on.Q2 = lowPass(d1.D12raw[select][3]/Fac2, LP=LP)
    off.I1 = lowPass(d2.D12raw[select][0]/Fac1, LP=LP)
    off.Q1 = lowPass(d2.D12raw[select][1]/Fac1, LP=LP)
    off.I2 = lowPass(d2.D12raw[select][2]/Fac2, LP=LP)
    off.Q2 = lowPass(d2.D12raw[select][3]/Fac2, LP=LP)
    dispData['I,Q data length'] = len(on.I1)
    dispData['Trace i, j, k'] = d1.ijk[select]


def lowPass(data, LP=0):
    if bool(LP):
        data = gaussian_filter1d(data, LP)  # Gausfilter
    return data


def makehist2d(dispData, dicData):
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    res = dicData['res']
    on.IImap, on.xII, on.yII = np.histogram2d(on.I1, on.I2, [on.xII, on.yII])
    on.QQmap, on.xQQ, on.yQQ = np.histogram2d(on.Q1, on.Q2, [on.xQQ, on.yQQ])
    on.IQmap, on.xIQ, on.yIQ = np.histogram2d(on.I1, on.Q2, [on.xIQ, on.yIQ])
    on.QImap, on.xQI, on.yQI = np.histogram2d(on.Q1, on.I2, [on.xQI, on.yQI])
    off.IImap, off.xII, off.yII = np.histogram2d(off.I1, off.I2, [on.xII, on.yII])
    off.QQmap, off.xQQ, off.yQQ = np.histogram2d(off.Q1, off.Q2, [on.xQQ, on.yQQ])
    off.IQmap, off.xIQ, off.yIQ = np.histogram2d(off.I1, off.Q2, [on.xIQ, on.yIQ])
    off.QImap, off.xQI, off.yQI = np.histogram2d(off.Q1, off.I2, [on.xQI, on.yQI])
    res.IQmapM[0] = on.IImap - off.IImap
    res.IQmapM[1] = on.QQmap - off.QQmap
    res.IQmapM[2] = on.IQmap - off.IQmap
    res.IQmapM[3] = on.QImap - off.QImap


def makeheader(dispData, dicData):
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    res = dicData['res']
    mapdim = dispData['mapdim']
    res.IQmapM = np.zeros([4, mapdim[0], mapdim[1]])
    res.IQmapM[0], on.xII, on.yII = np.histogram2d(on.I1, on.I2, mapdim)
    res.IQmapM[1], on.xQQ, on.yQQ = np.histogram2d(on.Q1, on.Q2, mapdim)
    res.IQmapM[2], on.xIQ, on.yIQ = np.histogram2d(on.I1, on.Q2, mapdim)
    res.IQmapM[3], on.xQI, on.yQI = np.histogram2d(on.Q1, on.I2, mapdim)
    on.headerII = ('Units,ufo,I1,' + str(on.xII[0]) + ',' + str(on.xII[-2]) +
                   ',I2,' + str(on.yII[0]) + ',' + str(on.yII[-2]) + ',DPow,2.03,0.03')
    on.headerQQ = ('Units,ufo,Q1,' + str(on.xQQ[0]) + ',' + str(on.xQQ[-2]) +
                   ',Q2,' + str(on.yQQ[0]) + ',' + str(on.yQQ[-2]) + ',DPow,2.03,0.03')
    on.headerIQ = ('Units,ufo,I1,' + str(on.xIQ[0]) + ',' + str(on.xIQ[-2]) +
                   ',Q2,' + str(on.yIQ[0]) + ',' + str(on.yIQ[-2]) + ',DPow,2.03,0.03')
    on.headerQI = ('Units,ufo,Q1,' + str(on.xQI[0]) + ',' + str(on.xQI[-2]) +
                   ',I2,' + str(on.yQI[0]) + ',' + str(on.yQI[-2]) + ',DPow,2.03,0.03')


def getCovMatrix(IQdata, lags=100, hp=True):
    # 0: <I1I2>
    # 1: <Q1Q2>
    # 2: <I1Q2>
    # 3: <Q1I2>
    # 4: <Squeezing> Magnitude
    # 5: <Squeezing> Phase
    lags = int(lags)
    I1 = np.asarray(IQdata[0])
    Q1 = np.asarray(IQdata[1])
    I2 = np.asarray(IQdata[2])
    Q2 = np.asarray(IQdata[3])
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
    rfftI2 = rfftn(I2[::-1], fshape)
    rfftQ2 = rfftn(Q2[::-1], fshape)
    if hp:
        # filter frequencies outside the lags range
        fftI1 = np.concatenate((np.zeros(HPfilt), fftI1[HPfilt:]))
        fftQ1 = np.concatenate((np.zeros(HPfilt), fftQ1[HPfilt:]))
        rfftI2 = np.concatenate((np.zeros(HPfilt), rfftI2[HPfilt:]))
        rfftQ2 = np.concatenate((np.zeros(HPfilt), rfftQ2[HPfilt:]))
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
    IQdata = np.array([on.I1, on.Q1, on.I2, on.Q2])
    num = 1
    if segment:
        num = len(IQdata[0])/segment
        modulo = len(IQdata[0]) % segment  # 4/3 = 1 + residual -> would give an error
        if modulo:
            for n in range(IQdata.shape[0]):
                IQdata[n] = IQdata[n][:-modulo]
        for n in range(IQdata.shape[0]):
            IQdata[n] = np.reshape(IQdata[n], [num, segment])
        CovMat = np.zeros([10, dispData['lags']*2+1])
        for i in range(num):
            CovMat += getCovMatrix(IQdata[:, i], lags=lags, hp=hp)
        CovMat = CovMat / np.float(num)
        CovMat[4, :] = abs(1j*(CovMat[2, :]+CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))
        CovMat[5, :] = np.angle(1j*(CovMat[2, :]+CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))
    else:
        CovMat = getCovMatrix(IQdata, lags=lags, hp=hp)
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
    res = dicData['res']
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
    res.CovMat = getCovMat_wrap(dispData, on)


def process(dispData, dicData):
    assignRaw(dispData, dicData)
    makeheader(dispData, dicData)
    get_data_avg(dispData, dicData)


def get_data_avg(dispData, dicData):
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    res = dicData['res']
    mapdim = dispData['mapdim']
    res.phn_avg = np.zeros(2)  # photon number
    res.IQmapM_avg = np.zeros([4, mapdim[0], mapdim[1]])  # histogram map
    res.c_avg = np.zeros([6, dispData['lags']*2+1])  # Covariance Map inc PSI
    for i in range(dispData['Averages']):
        assignRaw(dispData, dicData)
        logging.debug('Working on trace number '+str(dispData['Trace i, j, k']))
        logging.debug('dim1 value :' + str(dispData['dim1 lin'][int(dispData['Trace i, j, k'][0])]))
        correctPhase(dispData, dicData)  # assigned res.CovMat
        makehist2d(dispData, dicData)
        res.phn_avg[0] += np.mean(on.I1**2 + on.Q1**2) - np.mean(off.I1**2 + off.Q1**2)
        res.phn_avg[1] += np.mean(on.I2**2 + on.Q2**2) - np.mean(off.I2**2 + off.Q2**2)
        res.IQmapM_avg += res.IQmapM
        res.c_avg += res.CovMat
        dispData['select'] = dispData['select'] + 201  # for now a hard coded number!

    # res.IQmapM_avg = res.IQmapM_avg/dispData['Averages']
    res.phn_avg = res.phn_avg/dispData['Averages']
    res.c_avg = res.c_avg/dispData['Averages']
    res.c_avg[4, :] = abs(1j*(res.c_avg[2]+res.c_avg[3]) + (res.c_avg[0] - res.c_avg[1]))
    res.c_avg[5, :] = np.angle(1j*(res.c_avg[2]+res.c_avg[3]) + (res.c_avg[0] - res.c_avg[1]))
    res.c_avg[4, :] = 2.0*res.c_avg[4, :]/(res.phn_avg[0]+res.phn_avg[1]+1.0)

def process_all_points(dispData, dicData):
    assignRaw(dispData, dicData)
    makeheader(dispData, dicData)
    mapdim = dispData['mapdim']
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    start_select = dispData['select']
    pt = (dispData['dim1 pt'])
    res = dicData['res']
    res.IQmapMs_avg = np.zeros([pt, 4, mapdim[0], mapdim[1]])
    res.cs_avg = np.zeros([pt, 6, dispData['lags']*2+1])
    res.phns_avg = np.zeros([pt, 2])
    res.ineq_req = np.zeros(pt)
    f1 = np.float(dispData['f1'])
    f2 = np.float(dispData['f2'])

    for n in range(pt):
        get_data_avg(dispData, dicData)
        res.IQmapMs_avg[n] = res.IQmapM_avg
        res.cs_avg[n] = res.c_avg
        res.phns_avg[n] = res.phn_avg
        n1 = np.float(res.phn_avg[0])
        n2 = np.float(res.phn_avg[1])
        res.ineq_req[n] = (2.0*np.sqrt(f1*f2)*(n1+n2))/(f1*(2.0*n1+1.0)+f2*(2.0*n2+1.0))
        print res.ineq_req[n]
        dispData['select'] = start_select + 1 + n
