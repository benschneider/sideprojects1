import numpy as np
from multiprocessing import Process, Queue
from parsers import storehdf5  # , loadmtx, savemtx
from scipy.constants import h  # , e, c , pi
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
    Fac1 = np.sqrt(h * f1 * G1 * B)
    Fac2 = np.sqrt(h * f2 * G2 * B)
    return on, off, Fac1, Fac2


def assignRaw(dispData, dicData):
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    select = dispData['select']
    d1 = on.h5.root
    LP = dispData['Low Pass']
    d2 = off.h5.root
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    on.I1 = lowPass(d1.D12raw[select][0] / Fac1, lowpass_sigma=LP)
    on.Q1 = lowPass(d1.D12raw[select][1] / Fac1, lowpass_sigma=LP)
    on.I2 = lowPass(d1.D12raw[select][2] / Fac2, lowpass_sigma=LP)
    on.Q2 = lowPass(d1.D12raw[select][3] / Fac2, lowpass_sigma=LP)
    off.I1 = lowPass(d2.D12raw[select][0] / Fac1, lowpass_sigma=LP)
    off.Q1 = lowPass(d2.D12raw[select][1] / Fac1, lowpass_sigma=LP)
    off.I2 = lowPass(d2.D12raw[select][2] / Fac2, lowpass_sigma=LP)
    off.Q2 = lowPass(d2.D12raw[select][3] / Fac2, lowpass_sigma=LP)
    dispData['I,Q data length'] = len(on.I1)
    dispData['Trace i, j, k'] = d1.ijk[select]


def prep_data_hyb(dispData, dicData):
    on11 = dicData['on11']
    on22 = dicData['on22']
    on12 = dicData['on12']
    on21 = dicData['on21']
    off12 = dicData['off12']
    off21 = dicData['off21']
    f1 = dispData['f1']
    f2 = dispData['f2']
    G11 = dispData['Cross_correlation_Gain_f1f1']
    G22 = dispData['Cross_correlation_Gain_f2f2']
    B = dispData['B']
    Fac1 = np.sqrt(h * f1 * G11 * B)
    Fac2 = np.sqrt(h * f2 * G22 * B)
    dispData['IQ data length'] = on11.h5.root.D12raw.shape[2]
    dispData['maximum selectable'] = on11.h5.root.D12raw.shape[0]
    return on11, on22, on12, on21, off12, off21, Fac1, Fac2


def assignRaw_hyb(dispData, dicData):
    on11, on22, on12, on21, off12, off21, Fac1, Fac2 = prep_data_hyb(dispData, dicData)
    res = dispData['res']
    select = dispData['select']
    if select > dispData['maximum selectable']:
        logging.debug('selected trace num is than ' + str(dispData['maximum selectable']))
    LP =dispData['Low Pass']
    res.on_IQ = np.zeros([4, 4, dispData['IQ data length']])  # on11, 22, 12, 21
    res.off_IQ = np.zeros([2, 4, dispData['IQ data length']])  # off21, off12
    for n in range(4):
        res.on_IQ[0, n] = lowPass(on11.h5.root.D12raw[select][n] / Fac1, lowpass_sigma=dispData['Low Pass'])
        res.on_IQ[1, n] = lowPass(on12.h5.root.D12raw[select][n] / np.sqrt(Fac1*Fac2), lowpass_sigma=dispData['Low Pass'])
        res.on_IQ[2, n] = lowPass(on22.h5.root.D12raw[select][n] / Fac2, lowpass_sigma=dispData['Low Pass'])
        res.on_IQ[3, n] = lowPass(on21.h5.root.D12raw[select][n] / np.sqrt(Fac1*Fac2), lowpass_sigma=dispData['Low Pass'])
        res.off_IQ[0, n] = lowPass(off12.h.root.D12raw[select][n] / np.sqrt(Fac1*Fac2), lowpass_sigma=dispData['Low Pass'])
        res.off_IQ[1, n] = lowPass(off21.h.root.D12raw[select][n] / np.sqrt(Fac1*Fac2), lowpass_sigma=dispData['Low Pass'])
    dispData['Trace i, j, k'] = on11.h5.root.ijk[select]
    logging.debug('Assigned Hybrid coupler data trace ' + str(select) + ' to res object')


def makehist2d_hyb(res, mapdim):
    res.IQmapsON = np.zeros([4, mapdim[0], mapdim[1]])  # (4xON + 4xOFF) x 2D-Histogramms
    res.IQmapsOFF = np.zeros([4, mapdim[0], mapdim[1]])  # (4xON + 4xOFF) x 2D-Histogramms
    res.IQXbins = np.zeros([4, mapdim[0]+1])  # Bins are defined (from-to) therefore + 1
    res.IQYbins = np.zeros([4, mapdim[1]+1])  # Note : X and Y bins are shared between ON and OFF
    for i in range(2):
        # res.IQmapsON using on12 data: onI1I2, onQ1Q2, onI1Q2, onQ1I2 and OFF data respectively using off12 data
        res.IQmapsON[i+0], res.IQXbins[i+0], res.IQYbins[i+0] = np.histogram2d(res.on_IQ[1, i], res.on_IQ[1, i+2], [res.IQXbins[i], res.IQYbins[i]])
        res.IQmapsON[i+2], res.IQXbins[i+2], res.IQYbins[i+2] = np.histogram2d(res.on_IQ[1, i], res.on_IQ[1, 3-i], [res.IQXbins[i+2], res.IQYbins[i+2]])
        res.IQmapsOFF[i+0], res.IQXbins[i+0], res.IQYbins[i+0] = np.histogram2d(res.off_IQ[0, i], res.off_IQ[0, i+2], [res.IQXbins[i], res.IQYbins[i]])
        res.IQmapsOFF[i+2], res.IQXbins[i+2], res.IQYbins[i+2] = np.histogram2d(res.off_IQ[0, i], res.off_IQ[0, 3-i], [res.IQXbins[i+2], res.IQYbins[i+2]])
    res.IQmapsDiff = res.IQmapsON - res.IQmapsOFF


def makeheader_hyb(td, dicData):
    res = dicData['res']
    mapdim = td['mapdim']
    res.IQmaps = np.zeros([8, mapdim[0], mapdim[1]])  # (4xON + 4xOFF) x 2D-Histogramms
    res.IQXbins = np.zeros([8, mapdim[0]+1])  # Bins are defined (from-to) therefore + 1
    res.IQYbins = np.zeros([8, mapdim[1]+1])
    for i in range(2):
        # res.IQmapsON using on12 data: onI1I2, onQ1Q2, onI1Q2, onQ1I2 and OFF data respectively using off12 data
        res.IQmapsON[i+0], res.IQXbins[i+0], res.IQYbins[i+0] = np.histogram2d(res.on_IQ[1, i], res.on_IQ[1, i+2], mapdim)
        res.IQmapsON[i+2], res.IQXbins[i+2], res.IQYbins[i+2] = np.histogram2d(res.on_IQ[1, i], res.on_IQ[1, 3-i], mapdim)
        res.IQmapsOFF[i+0], res.IQXbins[i+0], res.IQYbins[i+0] = np.histogram2d(res.off_IQ[0, i], res.off_IQ[0, i+2], mapdim)
        res.IQmapsOFF[i+2], res.IQXbins[i+2], res.IQYbins[i+2] = np.histogram2d(res.off_IQ[0, i], res.off_IQ[0, 3-i], mapdim)
    header_string_dim3 = ',' + td['dim1 name'] + ',' +str(td['dim1 start']) + ',' +str(td['dim1 stop'])
    res.headerII = ('Units,Counts,I1,' + str(res.IQXbins[0][0]) + ',' + str(res.IQXbins[0][-2]) + ',I2,' + str(res.IQYbins[0][0]) + ',' + str(res.IQYbins[0][-2]) + header_string_dim3)
    res.headerQQ = ('Units,Counts,Q1,' + str(res.IQXbins[1][0]) + ',' + str(res.IQXbins[1][-2]) + ',Q2,' + str(res.IQYbins[1][0]) + ',' + str(res.IQYbins[1][-2]) + header_string_dim3)
    res.headerIQ = ('Units,Counts,I1,' + str(res.IQXbins[2][0]) + ',' + str(res.IQXbins[2][-2]) + ',Q2,' + str(res.IQYbins[2][0]) + ',' + str(res.IQYbins[2][-2]) + header_string_dim3)
    res.headerQI = ('Units,Counts,Q1,' + str(res.IQXbins[3][0]) + ',' + str(res.IQXbins[3][-2]) + ',I2,' + str(res.IQYbins[3][0]) + ',' + str(res.IQYbins[3][-2]) + header_string_dim3)

def process_hyb(disp_data, dic_data):
    res = dic_data['res']
    assignRaw_hyb(disp_data, dic_data)
    makeheader_hyb(disp_data, dic_data)
    get_data_hyb(res, dic_data)


def lowPass(data, lowpass_sigma=0.0):
    if bool(lowpass_sigma):
        data = gaussian_filter1d(data, lowpass_sigma)  # Gausfilter
    return data


def makehist2d(dispData, dicData):
    on, off, fac1, fac2 = prep_data(dispData, dicData)
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



def getCovMatrix(IQdata, lags=100, hp=False):
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
    CovMat = np.zeros([10, lags * 2 + 1])
    start = len(I1 * 2 - 1) - lags - 1
    stop = len(I1 * 2 - 1) + lags
    sI1 = np.array(I1.shape)
    sQ2 = np.array(Q2.shape)
    shape = sI1 + sQ2 - 1
    HPfilt = (int(sI1 / (lags * 8)))  # ignore features larger than (lags*8)
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
        fftQ2 = np.concatenate((np.zeros(HPfilt), fftQ1[HPfilt:]))
        fftI2 = np.concatenate((np.zeros(HPfilt), fftI1[HPfilt:]))
        fftQ1 = np.concatenate((np.zeros(HPfilt), fftQ1[HPfilt:]))
        rfftI1 = np.concatenate((np.zeros(HPfilt), rfftI2[HPfilt:]))
        rfftQ1 = np.concatenate((np.zeros(HPfilt), rfftQ2[HPfilt:]))
        rfftI2 = np.concatenate((np.zeros(HPfilt), rfftI2[HPfilt:]))
        rfftQ2 = np.concatenate((np.zeros(HPfilt), rfftQ2[HPfilt:]))
    CovMat[0, :] = (irfftn((fftI1 * rfftI2))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[1, :] = (irfftn((fftQ1 * rfftQ2))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[2, :] = (irfftn((fftI1 * rfftQ2))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[3, :] = (irfftn((fftQ1 * rfftI2))[fslice].copy()[start:stop] / len(fftI1))
    psi = (1j * (CovMat[2, :] + CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))
    CovMat[4, :] = abs(psi)
    CovMat[5, :] = np.angle(psi)
    CovMat[6, :] = (irfftn((fftI1 * rfftI1))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[7, :] = (irfftn((fftQ1 * rfftQ1))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[8, :] = (irfftn((fftI2 * rfftI2))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[9, :] = (irfftn((fftQ2 * rfftQ2))[fslice].copy()[start:stop] / len(fftI1))
    return CovMat


def getCovMat_wrap(dispData, data):
    segment = dispData['Segment Size']
    lags = dispData['lags']
    hp = dispData['FFT-Filter']
    IQdata = np.array([data.I1, data.Q1, data.I2, data.Q2])
    num = 1
    if segment:
        num = len(IQdata[0]) / segment
        modulo = len(IQdata[0]) % segment  # 4/3 = 1 + residual -> would give an error
        if bool(modulo):
            for n in range(IQdata.shape[0]):
                IQdata[n] = IQdata[n][:-modulo]
        # for n in range(IQdata.shape[0]):
        IQdata2 = np.reshape(IQdata[:], [IQdata.shape[0], num, segment])
        CovMat = np.zeros([10, dispData['lags'] * 2 + 1])
        for i in range(num):
            CovMat += getCovMatrix(IQdata2[:, i], lags=lags, hp=hp)
        CovMat = CovMat / np.float(num)
        psi = (1j * (CovMat[2, :] + CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))
        CovMat[4, :] = np.abs(psi)
        CovMat[5, :] = np.angle(psi)
    else:
        CovMat = getCovMatrix(IQdata, lags=lags, hp=hp)
    return CovMat


def f1pN2(tArray, lags0, d=1):
    squeezing_noise = np.sqrt(np.var(np.abs(tArray)))  # including the peak matters little
    if np.max(np.abs(tArray[lags0 - d:lags0 + d + 1])) < 3.5 * squeezing_noise:
        logging.debug('SN ratio too low: Can not find trigger position')
        distance = 0
    else:
        distance = (np.argmax(tArray[lags0 - d:lags0 + d + 1]) - d) * -1
    return distance


def f1pN(tArray, lags0, d=1):
    return (np.argmax(tArray[lags0 - d:lags0 + d + 1]) - d) * -1


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
        phase = np.angle(1j * on.Q1 + on.I1)  # phase rotation
        new = np.abs(1j * on.Q1 + on.I1) * np.exp(1j * (phase - phase_offset))
        on.I1 = np.real(new)
        on.Q1 = np.imag(new)
    on.CovMat = getCovMat_wrap(dispData, on)
    off.CovMat = getCovMat_wrap(dispData, off)


def process(dispData, dicData):
    assignRaw(dispData, dicData)
    makeheader(dispData, dicData)
    get_data_avg(dispData, dicData)


def get_data_hyb(res, disp_data):
    ''' Uses assigned data in res obj
    calculate cross correlations
    Trigger Jitters first
    Correct Phase rotation
    Create 4 Histogramms
    Obtain: Photon numbers, Squeezing, CovMatrix, Noise (sqrt(Variance)), Log Negativity
    '''
    lags = disp_data['lags']
    mapdim = disp_data['mapdim']
    res.n = np.zeros(2)
    # I1a = res.on12IQ[0]  # res.on21IQ second data set available
    covariancematrix = 1j*np.zeros([4, 4, lags*2+1])
    q = 4*[None]
    p = 4*[None]
    for i in range(4):
        # start calculation of submatrixes
        q[i] = Queue()
        p[i] = Process(target=get_covariance_submatrix(), args=(res.on_IQ[i], q[i]))
        p[i].start()

    timeout = 5
    for i in range(4):
        # collect calculated submatrixes
        p[i].join()
        covariancematrix[i] = q[i].get(True, timeout)



def get_covariance_submatrix(IQdata, q, hp=False):
    logging.debug('Calculating Submatrix')
    # return submatrix: 0: <I1I2>  1: <Q1Q2>  2: <I1Q2>  3: <Q1I2>
    I1 = IQdata[0]
    Q1 = IQdata[1]
    I2 = IQdata[2]
    Q2 = IQdata[3]
    lags = int(lags)
    CovMat = np.zeros([4, lags * 2 + 1])
    sI1 = np.array(I1.shape)
    sQ2 = np.array(Q2.shape)
    shape = sI1 + sQ2 - 1
    HPfilt = (int(sI1 / (lags * 8)))  # ignore features larger than (lags*8)
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
        fftQ2 = np.concatenate((np.zeros(HPfilt), fftQ1[HPfilt:]))
        fftI2 = np.concatenate((np.zeros(HPfilt), fftI1[HPfilt:]))
        fftQ1 = np.concatenate((np.zeros(HPfilt), fftQ1[HPfilt:]))
        rfftI1 = np.concatenate((np.zeros(HPfilt), rfftI2[HPfilt:]))
        rfftQ1 = np.concatenate((np.zeros(HPfilt), rfftQ2[HPfilt:]))
        rfftI2 = np.concatenate((np.zeros(HPfilt), rfftI2[HPfilt:]))
        rfftQ2 = np.concatenate((np.zeros(HPfilt), rfftQ2[HPfilt:]))
    CovMat[0, :] = (irfftn((fftI1 * rfftI2))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[1, :] = (irfftn((fftQ1 * rfftQ2))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[2, :] = (irfftn((fftI1 * rfftQ2))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[3, :] = (irfftn((fftQ1 * rfftI2))[fslice].copy()[start:stop] / len(fftI1))
    psi = (1j * (CovMat[2, :] + CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))
    CovMat[4, :] = abs(psi)
    CovMat[5, :] = np.angle(psi)
    CovMat[6, :] = (irfftn((fftI1 * rfftI1))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[7, :] = (irfftn((fftQ1 * rfftQ1))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[8, :] = (irfftn((fftI2 * rfftI2))[fslice].copy()[start:stop] / len(fftI1))
    CovMat[9, :] = (irfftn((fftQ2 * rfftQ2))[fslice].copy()[start:stop] / len(fftI1))
    return CovMat


def get_data_avg(dispData, dicData):
    dd = dispData
    on, off, Fac1, Fac2 = prep_data(dd, dicData)
    res = dicData['res']
    lags = dd['lags']
    mapdim = dd['mapdim']
    res.n = np.zeros(2)  # photon number
    res.IQmapM_avg = np.zeros([4, mapdim[0], mapdim[1]])  # histogram map
    res.c_avg = np.zeros([10, lags * 2 + 1])  # Covariance Map inc PSI
    res.c_avg_off = np.zeros([10, lags * 2 + 1])  # Covariance Map
    res.psi_avg = 1j * np.zeros([1, lags * 2 + 1])  # PSI
    for i in range(dd['Averages']):
        assignRaw(dd, dicData)
        logging.debug('Working on trace number ' + str(dd['Trace i, j, k']))
        logging.debug('dim1 value :' + str(dd['dim1 lin'][int(dd['Trace i, j, k'][0])]))
        correctPhase(dd, dicData)  # assigns res.CovMat
        makehist2d(dd, dicData)
        res.IQmapM_avg += res.IQmapM
        res.c_avg += on.CovMat
        res.c_avg_off += off.CovMat
        res.n[0] += (on.CovMat[6] + on.CovMat[7] - off.CovMat[6] - off.CovMat[7])[lags]
        res.n[1] += (on.CovMat[8] + on.CovMat[9] - off.CovMat[8] - off.CovMat[9])[lags]
        dd['select'] = dd['select'] + 201  # for now a hard coded number!

    res.n = 0.5 + np.abs(res.n) / dd['Averages']  # force averaged value to be larger than 0.5
    res.c_avg_off = res.c_avg_off / dd['Averages']
    res.c_avg = res.c_avg / dd['Averages']
    res.psi_avg[0, :] = (res.c_avg[0] * 1.0 - res.c_avg[1] * 1.0 +
                         1j * (res.c_avg[2] * 1.0 + res.c_avg[3] * 1.0))
    res.psi_avg[0, :] = res.psi_avg[0, :] - np.mean(res.psi_avg[0, 0:lags - 10])  # remove offset

    res.sq, res.ineq, res.noise = get_sq_ineq(res.psi_avg[0],
                                              res.n[0],
                                              res.n[1],
                                              np.float(dd['f1']),
                                              np.float(dd['f2']),
                                              lags)
    res.sqph = np.angle(res.psi_avg[0][lags])


def get_sq_ineq(psi, n1, n2, f1, f2, lags):
    '''returns the ammount of squeezing, ineq and noise'''
    noise = np.sqrt(np.var(np.abs(psi)))
    logging.debug('Mag Psi sqrt(Variance): ' + str(noise))
    squeezing = np.max(np.abs(psi)) / ((n1 + n2) / 2.0)  # w. zpf
    logging.debug(('n1: ' + str(n1) + ' n2: ' + str(n2)))
    a = 2.0 * np.sqrt(f1 * f2) * np.abs(n1 + n2 - 1)
    b = f1 * (2.0 * n1 + 1.0 - 0.5) + f2 * (2.0 * n2 + 1.0 - 0.5)
    ineq = a / b   # does not include zpf
    logging.debug(('ineq: ' + str(ineq) + ' sq: ' + str(squeezing)))
    return squeezing, ineq, noise


def process_all_points(dispData, dicData):
    assignRaw(dispData, dicData)
    makeheader(dispData, dicData)
    lags = dispData['lags']
    mapdim = dispData['mapdim']
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    start_select = dispData['select']
    pt = (dispData['dim1 pt'])
    res = dicData['res']
    res.IQmapMs_avg = np.zeros([pt, 4, mapdim[0], mapdim[1]])
    res.cs_avg = np.zeros([pt, 10, lags * 2 + 1])
    res.cs_avg_off = np.zeros([pt, 10, lags * 2 + 1])
    res.ns = np.zeros([pt, 2])
    res.ineqs = np.zeros(pt)
    res.sqs = np.zeros(pt)
    res.sqphs = np.zeros(pt)
    res.noises = np.zeros(pt)
    for n in range(pt):
        get_data_avg(dispData, dicData)
        res.IQmapMs_avg[n] = res.IQmapM_avg
        res.cs_avg[n] = res.c_avg
        res.cs_avg_off[n] = res.c_avg_off
        res.ns[n] = res.n
        res.sqs[n] = res.sq
        res.sqphs[n] = res.sqph
        res.ineqs[n] = res.ineq
        res.noises[n] = res.noise
        print 'SQ:', str(res.sq), 'INEQ:', str(res.ineq)
        dispData['select'] = start_select + n + 1

if __name__ == '__main__':
    # testing individual functions
    pass