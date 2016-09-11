import numpy as np
from parsers import storehdf5  # , loadmtx, savemtx
from scipy.constants import h  # , e, c , pi
from scipy.signal.signaltools import _next_regular
from scipy.ndimage.filters import gaussian_filter1d
from numpy.fft import rfftn, irfftn
import logging
from multiprocessing import Process, Queue, Pipe
from time import time
import PyGnuplot as gp
import gc

def load_dataset(dispData, dicData, ext='hdf5_on'):
    '''Reads from the dictionary which data set was selected'''
    dicData[ext] = storehdf5(str(dispData[ext]))
    dicData[ext].open_f(mode='r')


def f1pN2(tArray, lags0, d=1):
    squeezing_noise = np.sqrt(np.var(np.abs(tArray)))  # including the peak matters little
    if np.max(np.abs(tArray[lags0 - d:lags0 + d + 1])) < 2.5 * squeezing_noise:
        logging.debug('SN ratio too low: Can not find trigger position')
        distance = 0
    else:
        distance = (np.argmax(tArray[lags0 - d:lags0 + d + 1]) - d) * -1
    return distance


def f1pN(tArray, lags0, d=1):
    return (np.argmax(tArray[lags0 - d:lags0 + d + 1]) - d) * -1


def process(dispData, dicData):
    assignRaw(dispData, dicData)
    makeheader(dispData, dicData)
    get_data_avg(dispData, dicData)


def process_all_points(dispData, dicData):
    assignRaw(dispData, dicData)
    makeheader(dispData, dicData)
    lags = dispData['lags']
    mapdim = dispData['mapdim']
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    start_select = dispData['select']
    pt = int(dispData['Process Num'])
    res = dicData['res']
    res.IQmapMs_avg = np.zeros([pt, 4, mapdim[0], mapdim[1]])
    res.cs_avg = np.zeros([pt, 10, lags * 2 + 1])
    res.cs_avg_off = np.zeros([pt, 10, lags * 2 + 1])
    res.ns = np.zeros([pt, 2])
    res.ineqs = np.zeros(pt)
    res.sqs = np.zeros(pt)
    res.sqs2 = np.zeros(pt)
    res.sqsn2 = np.zeros(pt)
    res.sqphs = np.zeros(pt)
    res.noises = np.zeros(pt)
    for n in range(pt):
        get_data_avg(dispData, dicData)
        res.IQmapMs_avg[n] = res.IQmapM_avg
        res.cs_avg[n] = res.c_avg
        res.cs_avg_off[n] = res.c_avg_off
        res.ns[n] = res.n
        res.sqs[n] = res.sq
        res.sqs2[n] = res.psi_mag_avg[0]
        res.sqsn2[n] = res.psi_mag_avg[1]
        res.sqphs[n] = res.sqph
        res.ineqs[n] = res.ineq
        res.noises[n] = res.noise
        print 'SQ:', str(res.sq), 'INEQ:', str(res.ineq)
        dispData['select'] = start_select + n + 1


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
    dispData['I,Q data length'] = int(len(on.I1))
    dispData['Trace i, j, k'] = list(d1.ijk[select])


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


def lowPass(data, lowpass_sigma=0.0):
    if bool(lowpass_sigma):
        data = gaussian_filter1d(data, lowpass_sigma)  # Gausfilter
    return data - np.mean(data)  # remove DC offset


def mp_wrap(pin, function, *args2):
    ''' retvalues are the returns  and function is the function that gives you these'''
    retvalues = function(*args2)
    pin.send(retvalues)
    pin.close()


def makehist2d(dispData, dicData):
    gc.collect()
    on, off, fac1, fac2 = prep_data(dispData, dicData)
    t0 = time()
    res = dicData['res']
    # pout = 8*[None]
    # pin = 8*[None]
    # p = 8*[None]
    # for i in range(8):
    #     pout[i], pin[i] = Pipe()  # This crashes way to easy!!!
    # p[0] = Process(target=mp_wrap, args=(pin[0], np.histogram2d, on.I1, on.I2, [on.xII, on.yII]))
    # p[1] = Process(target=mp_wrap, args=(pin[1], np.histogram2d, on.Q1, on.Q2, [on.xQQ, on.yQQ]))
    # p[2] = Process(target=mp_wrap, args=(pin[2], np.histogram2d, on.I1, on.Q2, [on.xIQ, on.yIQ]))
    # p[3] = Process(target=mp_wrap, args=(pin[3], np.histogram2d, on.Q1, on.I2, [on.xQI, on.yQI]))
    # p[4] = Process(target=mp_wrap, args=(pin[4], np.histogram2d, off.I1, off.I2, [on.xII, on.yII]))
    # p[5] = Process(target=mp_wrap, args=(pin[5], np.histogram2d, off.Q1, off.Q2, [on.xQQ, on.yQQ]))
    # p[6] = Process(target=mp_wrap, args=(pin[6], np.histogram2d, off.I1, off.Q2, [on.xIQ, on.yIQ]))
    # p[7] = Process(target=mp_wrap, args=(pin[7], np.histogram2d, off.Q1, off.I2, [on.xQI, on.yQI]))
    # for i in range(8):
    #     p[i].start()
    #     pin[i].close()
    # for i in range(8):
    #     p[i].join()
    # on.IImap, on.xII, on.yII = pout[0].recv()
    # on.QQmap, on.xQQ, on.yQQ = pout[1].recv()
    # on.IQmap, on.xIQ, on.yIQ = pout[2].recv()
    # on.QImap, on.xQI, on.yQI = pout[3].recv()
    # off.IImap, off.xII, off.yII = pout[4].recv()
    # off.QQmap, off.xQQ, off.yQQ = pout[5].recv()
    # off.IQmap, off.xIQ, off.yIQ = pout[6].recv()
    # off.QImap, off.xQI, off.yQI = pout[7].recv()
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
    t1 = time()
    logging.debug('End MP histogramm time used: ' + str(t1-t0))


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
    # 0: <I1I2> # 1: <Q1Q2> # 2: <I1Q2> # 3: <Q1I2> # 4: <Squeezing> Magnitude # 5: <Squeezing> Phase
    lags = int(lags)
    I1 = np.asarray(IQdata[0])
    Q1 = np.asarray(IQdata[1])
    I2 = np.asarray(IQdata[2])
    Q2 = np.asarray(IQdata[3])
    CovMat = np.zeros([10, lags * 2 + 1])
    start = len(I1) - lags - 1
    stop = len(I1) + lags
    sI1 = np.array(I1.shape)
    shape0 = sI1*2 - 1
    fshape = [_next_regular(int(d)) for d in shape0]  # padding to optimal size for FFTPACK
    fslice = tuple([slice(0, int(sz)) for sz in shape0])
    # Do FFTs and get Cov Matrix
    fftI1 = rfftn(I1, fshape)
    fftQ1 = rfftn(Q1, fshape)
    fftI2 = rfftn(I2, fshape)
    fftQ2 = rfftn(Q2, fshape)
    rfftI1 = rfftn(I1[::-1], fshape)  # there should be a simple relationship to fftI1
    rfftQ1 = rfftn(Q1[::-1], fshape)
    rfftI2 = rfftn(I2[::-1], fshape)
    rfftQ2 = rfftn(Q2[::-1], fshape)
    # if hp:
    #     # filter frequencies outside the lags range
    # HPfilt = (int(sI1 / (lags * 8)))  # ignore features larger than (lags*8)
    # fftI1 = np.concatenate((np.zeros(HPfilt), fftI1[HPfilt:]))
    # fftQ2 = np.concatenate((np.zeros(HPfilt), fftQ1[HPfilt:]))
    # fftI2 = np.concatenate((np.zeros(HPfilt), fftI1[HPfilt:]))
    # fftQ1 = np.concatenate((np.zeros(HPfilt), fftQ1[HPfilt:]))
    # rfftI1 = np.concatenate((np.zeros(HPfilt), rfftI2[HPfilt:]))
    # rfftQ1 = np.concatenate((np.zeros(HPfilt), rfftQ2[HPfilt:]))
    # rfftI2 = np.concatenate((np.zeros(HPfilt), rfftI2[HPfilt:]))
    # rfftQ2 = np.concatenate((np.zeros(HPfilt), rfftQ2[HPfilt:]))
    #
    CovMat[0, :] = irfftn((fftI1 * rfftI2))[fslice].copy()[start:stop]/fshape
    CovMat[1, :] = irfftn((fftQ1 * rfftQ2))[fslice].copy()[start:stop]/fshape
    CovMat[2, :] = irfftn((fftI1 * rfftQ2))[fslice].copy()[start:stop]/fshape
    CovMat[3, :] = irfftn((fftQ1 * rfftI2))[fslice].copy()[start:stop]/fshape
    psi = (1j * (CovMat[2, :] + CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))
    CovMat[4, :] = abs(psi)
    CovMat[5, :] = np.angle(psi)
    CovMat[6, :] = irfftn((fftI1 * rfftI1))[fslice].copy()[start:stop]/fshape
    CovMat[7, :] = irfftn((fftQ1 * rfftQ1))[fslice].copy()[start:stop]/fshape
    CovMat[8, :] = irfftn((fftI2 * rfftI2))[fslice].copy()[start:stop]/fshape
    CovMat[9, :] = irfftn((fftQ2 * rfftQ2))[fslice].copy()[start:stop]/fshape
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


def correctPhase(dispData, dicData):
    on, off, Fac1, Fac2 = prep_data(dispData, dicData)
    if dispData['Trigger correction']:
        CovMat = getCovMat_wrap(dispData, on)
        dMag = f1pN2(CovMat[4], dispData['lags'], d=1)
        on.I1 = np.roll(on.I1, dMag)  # Correct 1pt trigger jitter
        on.Q1 = np.roll(on.Q1, dMag)
        logging.debug('Trigger corrected ' + str(dMag) + 'pt')

    if dispData['Phase correction']:
        phase_index = np.argmax(CovMat[4])
        phase_offset = CovMat[5][phase_index]
        phase = np.angle(1j * on.Q1 + on.I1)  # phase rotation
        new = np.abs(1j * on.Q1 + on.I1) * np.exp(1j * (phase - phase_offset))
        on.I1 = np.real(new)
        on.Q1 = np.imag(new)
        logging.debug('Phase corrected ' + str(phase_offset) + 'rad')

    on.CovMat = CovMat  # getCovMat_wrap(dispData, on)
    # off.CovMat = getCovMat_wrap(dispData, off)


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
    res.psi_mag_avg = np.zeros(2)
    covMat = np.zeros([4,4])
    for i in range(dd['Averages']):
        assignRaw(dd, dicData)
        logging.debug('Working on trace number ' + str(dd['Trace i, j, k']))
        logging.debug('dim1 value :' + str(dicData['dim1 lin'][int(dd['Trace i, j, k'][0])]))
        correctPhase(dd, dicData)  # assigns res.CovMat
        makehist2d(dd, dicData)
        res.IQmapM_avg += res.IQmapM
        res.c_avg += on.CovMat
        dd['select'] = dd['select'] + 201  # for now a hard coded number!
        covMat0 = np.cov([on.I1, on.Q1, on.I2, on.Q2])
        covMat1 = np.cov([off.I1, off.Q1, off.I2, off.Q2])
        covMat += covMat0
        covMat[0, 0] -= covMat1[0, 0]
        covMat[1, 1] -= covMat1[1, 1]
        covMat[2, 2] -= covMat1[2, 2]
        covMat[3, 3] -= covMat1[3, 3]
        res.psi_mag_avg[0] += np.abs((covMat0[2, 0] - covMat0[3, 1]) + 1j*(covMat0[3, 0] + covMat0[2, 1]))
        res.psi_mag_avg[1] += np.abs((covMat1[2, 0] - covMat1[3, 1]) + 1j*(covMat1[3, 0] + covMat1[2, 1]))

    res.psi_mag_avg /= dd['Averages']
    res.n[0] = (covMat[0, 0] + covMat[1, 1])
    res.n[1] = (covMat[2, 2] + covMat[3, 3])
    res.covMat = covMat/dd['Averages']
    res.n = 0.5 + res.n / dd['Averages']  # force averaged value to be larger than 0.5
    # res.c_avg_off = res.c_avg_off / dd['Averages']
    res.c_avg = res.c_avg / dd['Averages']
    res.psi = (res.covMat[2, 0] - res.covMat[3, 1]) + 1j*(res.covMat[3, 0] + res.covMat[2, 1])
    res.psi_avg[0, :] = (res.c_avg[0] * 1.0 - res.c_avg[1] * 1.0 + 1j * (res.c_avg[2] * 1.0 + res.c_avg[3] * 1.0))
    # res.psi_avg[0, :] = (res.c_avg[0] * 1.0 - res.c_avg[1] * 1.0 + 1j * (res.c_avg[2] * 1.0 + res.c_avg[3] * 1.0))
    # res.psi_avg[0, :] = res.psi_avg[0, :] - np.mean(res.psi_avg[0, 0:lags - 10])  # remove offset (shouldn't do)
    res.sq, res.ineq, res.noise = get_sq_ineq(res.psi,  # res.psi_avg[0],
                                              res.n[0],
                                              res.n[1],
                                              np.float(dd['f1']),
                                              np.float(dd['f2']),
                                              lags)
    # res.sq = res.psi_mag_avg[0]
    logging.debug(str(res.psi_mag_avg))
    logging.debug('n1full: ' + str(np.mean(on.I1**2+on.Q1**2-off.I1**2-off.Q1**2) ) )
    logging.debug('On Matrix:'+str(res.covMat))
    res.sqph = np.angle(res.psi_avg[0][lags])


def get_sq_ineq(psi, n1, n2, f1, f2, lags):
    '''returns the ammount of squeezing, ineq and noise'''
    noise = np.sqrt(np.var(np.abs(psi)))
    logging.debug('Mag Psi sqrt(Variance): ' + str(noise))
    squeezing = np.max(np.abs(psi)) # / ((n1 + n2) / 2.0)  # w. zpf
    logging.debug(('n1: ' + str(n1) + ' n2: ' + str(n2)))
    a = 2.0 * np.sqrt(f1 * f2) * np.abs(n1 + n2 - 1)
    b = f1 * (2.0 * n1 + 1.0 - 0.5) + f2 * (2.0 * n2 + 1.0 - 0.5)
    ineq = a / b   # does not include zpf
    logging.debug(('ineq: ' + str(ineq) + ' sq raw: ' + str(squeezing)))
    return squeezing, ineq, noise
