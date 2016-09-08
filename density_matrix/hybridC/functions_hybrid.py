from multiprocessing import Process, Queue, Pipe
import numpy as np
import logging
from scipy.constants import h
from scipy.signal.signaltools import _next_regular
from scipy.ndimage.filters import gaussian_filter1d
from numpy.fft import rfftn, irfftn


def prep_data_hyb(dispData, dicData):
    select = dispData['select']
    on11 = dicData['on11']
    on22 = dicData['on22']
    on12 = dicData['on12']
    on21 = dicData['on21']
    off12 = dicData['off12']
    off21 = dicData['off21']
    f1 = dispData['f1']
    f2 = dispData['f2']
    Gain11 = np.linspace(dispData['cgain11 stop'], dispData['cgain11 start'], dispData['dim1 pt'])
    Gain22 = np.linspace(dispData['cgain22 stop'], dispData['cgain22 start'], dispData['dim1 pt'])
    Gab = Gain11[select]
    Gba = Gain22[select]
    logging.debug('G11 '+str(Gab))
    logging.debug('G22 '+str(Gba))
    B = dispData['B']
    Fac1 = np.sqrt(h * f1 * Gab * B)
    Fac2 = np.sqrt(h * f2 * Gba * B)
    dispData['IQ data length'] = on11.h5.root.D12raw.shape[2]
    dispData['maximum selectable'] = on11.h5.root.D12raw.shape[0]
    return on11, on22, on12, on21, off12, off21, Fac1, Fac2


def assignRaw_hyb(dispData, dicData):
    on11, on22, on12, on21, off12, off21, Fac1, Fac2 = prep_data_hyb(dispData, dicData)
    res = dicData['res']
    select = dispData['select']
    if select > dispData['maximum selectable']:
        logging.debug('selected trace num is than ' + str(dispData['maximum selectable']))
    LP =dispData['Low Pass']
    res.on_IQ = np.zeros([4, 4, dispData['IQ data length']])  # on11, 12, 21, 22
    res.off_IQ = np.zeros([2, 4, dispData['IQ data length']])  # off21, off12
    ncorr = np.ones(4)
    if dispData['drift']:
        ncorr[0] = res.cpow1a[select]
        ncorr[1] = res.cpow1b[select]
        ncorr[2] = res.cpow2a[select]
        ncorr[3] = res.cpow2b[select]
    for n in range(4):
        res.on_IQ[0, n] = lowPass(on11.h5.root.D12raw[select][n] / Fac1 / np.sqrt(ncorr[0]*ncorr[1]), lowpass_sigma=LP)
        res.on_IQ[1, n] = lowPass(on12.h5.root.D12raw[select][n] / np.sqrt(Fac1*Fac2) / np.sqrt(ncorr[0]*ncorr[3]), lowpass_sigma=LP)
        res.on_IQ[2, n] = lowPass(on21.h5.root.D12raw[select][n] / np.sqrt(Fac1*Fac2) / np.sqrt(ncorr[2]*ncorr[1]), lowpass_sigma=LP)
        res.on_IQ[3, n] = lowPass(on22.h5.root.D12raw[select][n] / Fac2 / np.sqrt(ncorr[2]*ncorr[3]), lowpass_sigma=LP)
        res.off_IQ[0, n] = lowPass(off12.h5.root.D12raw[select][n] / np.sqrt(Fac1*Fac2), lowpass_sigma=LP)
        res.off_IQ[1, n] = lowPass(off21.h5.root.D12raw[select][n] / np.sqrt(Fac1*Fac2), lowpass_sigma=LP)
    dispData['Trace i, j, k'] = on11.h5.root.ijk[select]
    logging.debug('Assigned Hybrid coupler data trace ' + str(select) + ' to res object')


def process_hyb(disp_data, dic_data):
    res = dic_data['res']
    assignRaw_hyb(disp_data, dic_data)
    makeheader_hyb(disp_data, dic_data)  # creates histograms
    get_data_hyb(res, disp_data)
    lags = disp_data['lags']
    # res.cov_mat = res.cov_matrix_ON[:, lags]
    res.cov_mat = np.zeros([4, 4, lags*2+1])
    res.cov_mat[0, 0:2] = res.cov_matrix_ON[0:2]    # On11 <II> <IQ>
    res.cov_mat[1, 0:2] = res.cov_matrix_ON[2:4]    # On11 <QI> <QQ>
    res.cov_mat[0, 2:4] = res.cov_matrix_ON[4:6]    # On12 <II> <IQ>
    res.cov_mat[1, 2:4] = res.cov_matrix_ON[6:8]    # On12 <QI> <QQ>
    res.cov_mat[2, 0:2] = res.cov_matrix_ON[8:10]   # On21 <II> <IQ>
    res.cov_mat[3, 0:2] = res.cov_matrix_ON[10:12]  # On21 <QI> <QQ>
    res.cov_mat[2, 2:4] = res.cov_matrix_ON[12:14]  # On22 <II> <IQ>
    res.cov_mat[3, 2:4] = res.cov_matrix_ON[14:16]  # On22 <QI> <QQ>
    res.cov_mat = res.cov_mat[:, :, lags].flatten()


def process_hyb2(disp_data, dic_data):
    res = dic_data['res']
    lags = disp_data['lags']
    assignRaw_hyb(disp_data, dic_data)
    makeheader_hyb(disp_data, dic_data)  # creates histograms
    get_data_hyb2(res, disp_data)
    res.cov_mat = np.zeros([8, 8, (lags*2+1)])
    res.cov_mat[0, 0:4] = res.cov_matrix_ON[0:4]        # On11 <I1I1> <IQ>
    res.cov_mat[1, 0:4] = res.cov_matrix_ON[4:8]        # On11 <QI> <QQ>
    res.cov_mat[2, 0:4] = res.cov_matrix_ON[8:12]       # On11 <II> <IQ>
    res.cov_mat[3, 0:4] = res.cov_matrix_ON[12:16]      # On11 <QI> <QQ>
    res.cov_mat[0, 4:8] = res.cov_matrix_ON[16:20]      # On12 <II> <IQ>
    res.cov_mat[1, 4:8] = res.cov_matrix_ON[20:24]      # On12 <QI> <QQ>
    res.cov_mat[2, 4:8] = res.cov_matrix_ON[24:28]      # On12 <II> <IQ>
    res.cov_mat[3, 4:8] = res.cov_matrix_ON[28:32]      # On12 <QI> <QQ>
    res.cov_mat[4, 0:4] = res.cov_matrix_ON[32:36]      # On21 <II> <IQ>
    res.cov_mat[5, 0:4] = res.cov_matrix_ON[36:40]      # On21 <QI> <QQ>
    res.cov_mat[6, 0:4] = res.cov_matrix_ON[40:44]      # On21 <II> <IQ>
    res.cov_mat[7, 0:4] = res.cov_matrix_ON[44:48]      # On21 <QI> <QQ>
    res.cov_mat[4, 4:8] = res.cov_matrix_ON[48:52]      # On22 <II> <IQ>
    res.cov_mat[5, 4:8] = res.cov_matrix_ON[52:56]      # On22 <QI> <QQ>
    res.cov_mat[6, 4:8] = res.cov_matrix_ON[56:60]      # On22 <II> <IQ>
    res.cov_mat[7, 4:8] = res.cov_matrix_ON[60:64]      # On22 <QI> <QQ>
    res.cov_mat = res.cov_mat[:, :, lags]


def process_hyb_all(disp_data, dic_data):
    res = dic_data['res']
    for i in range(disp_data['Process Num']):
        disp_data['select'] = i
        assignRaw_hyb(disp_data, dic_data)
        # makeheader_hyb(disp_data, dic_data)  # creates histograms
        # TODO Histograms plus headers
        get_data_hyb(res, disp_data)
        lags = disp_data['lags']


def get_data_hyb(res, disp_data):
    ''' Uses assigned data in res obj
    calculate cross correlations
    Trigger Jitters first
    Correct Phase rotation
    Create 4 Histogramms
    Obtain: Photon numbers, Squeezing, CovMatrix, Noise (sqrt(Variance)), Log Negativity
    '''
    lags = disp_data['lags']
    res.n = np.zeros(2)
    covariance_matrix = np.zeros([16, lags*2+1])
    q = 4*[None]
    p = 4*[None]
    timeout = 5
    for i in range(4):
        q[i] = Queue()
        p[i] = Process(name=('Process '+str(i)), target=get_covariance_submatrix, args=(res.on_IQ[i], lags, q[i]))
        p[i].start()
    for i in range(4):
        p[i].join()  # wait for calculation to complete

    phasefix = disp_data['Phase correction']
    trigfix = disp_data['Trigger correction']
    covariance_matrix[:4] = get_corr_cov_mat(q[0].get(True, timeout), lags, triggerfix=trigfix, phasefix=phasefix, nbasis=True)
    covariance_matrix[4:8] = get_corr_cov_mat(q[1].get(True, timeout), lags, triggerfix=trigfix, phasefix=phasefix, nbasis=False)
    covariance_matrix[8:12] = get_corr_cov_mat(q[2].get(True, timeout), lags, triggerfix=trigfix, phasefix=phasefix, nbasis=False)
    covariance_matrix[12:16] = get_corr_cov_mat(q[3].get(True, timeout), lags, triggerfix=trigfix, phasefix=phasefix, nbasis=True)

    # Assemble full matrix
    res.cov_mat = np.zeros([4, 4, lags*2+1])

    # make histogram symetric
    if abs(covariance_matrix[4]) > abs(covariance_matrix[8]):
        a1 = covariance_matrix[4:6]
        b1 = covariance_matrix[6:8]
    else:
        a1 = covariance_matrix[8:10]
        b1 = covariance_matrix[10:12]
    res.cov_mat[0, 0:2] = covariance_matrix[0:2]    # On11 <II> <IQ> Top left
    res.cov_mat[1, 0:2] = covariance_matrix[2:4]    # On11 <QI> <QQ>
    res.cov_mat[0, 2:4] = a1  # covariance_matrix[4:6]    # On12 <II> <IQ> Top right
    res.cov_mat[1, 2:4] = b1  # covariance_matrix[6:8]    # On12 <QI> <QQ>
    res.cov_mat[2, 0:2] = a1  # covariance_matrix[8:10]   # On21 <II> <IQ> Bottom left
    res.cov_mat[3, 0:2] = b1  # covariance_matrix[10:12]  # On21 <QI> <QQ>
    res.cov_mat[2, 2:4] = covariance_matrix[12:14]  # On22 <II> <IQ> Bottom right
    res.cov_mat[3, 2:4] = covariance_matrix[14:16]  # On22 <QI> <QQ>
    res.cov_mat = res.cov_mat[:, :, lags].flatten()

def get_data_hyb2(res, disp_data):
    ''' Uses assigned data in res obj
    calculate cross correlations
    Trigger Jitters first
    Correct Phase rotation
    Create 4 Histogramms
    Obtain: Photon numbers, Squeezing, CovMatrix, Noise (sqrt(Variance)), Log Negativity
    '''
    lags = disp_data['lags']
    res.n = np.zeros(2)
    covariance_matrix = np.zeros([16*4, lags*2+1])

    # Doing Sequencial processing: Passing big data via pipe tends to be unstable
    covariance_matrix[0:16] = get_covariance_submatrix_full(res.on_IQ[0], lags)
    covariance_matrix[16:32] = get_covariance_submatrix_full(res.on_IQ[1], lags)
    covariance_matrix[32:48] = get_covariance_submatrix_full(res.on_IQ[2], lags)
    covariance_matrix[48:64] = get_covariance_submatrix_full(res.on_IQ[3], lags)
    # phasefix = disp_data['Phase correction']
    # trigfix = disp_data['Trigger correction']
    res.cov_matrix_ON = covariance_matrix


def get_corr_cov_mat(IQdata, lags, triggerfix=True, phasefix=True, nbasis=False):
    IIc = IQdata[0]
    IQc = IQdata[1]
    QIc = IQdata[2]
    QQc = IQdata[3]
    # Fix Phaseflip (always active)
    phaseflip = False
    psi = (IIc - QQc + 1j*(IQc + QIc))
    psi2 = (QIc - IQc + 1j*(QQc + IIc))
    if np.max(np.abs(psi)) < np.max(np.abs(psi2)):
        # phaseflip check (more reliable?)
        phaseflip = True
        psi = psi2
    if triggerfix:
        # Fix Trigger jitter
        dist = (np.argmax(np.abs(psi)[(lags - 1):(lags + 2)]) - 1) * -1
        logging.debug('Trigger jitter distance: ' + str(dist))
        psi = np.roll(psi, dist)
        dist = (np.argmax(np.abs(psi)[(lags - 1):(lags + 2)]) - 1) * -1
        logging.debug('Corrected jitter distance: ' + str(dist))
    if phasefix:
        # Phase Correction
        phase_offset = np.angle(psi)[lags]
        logging.debug('found phase offset: ' + str(phase_offset))
        psi = psi * np.exp(1j * -1 * phase_offset)
    if np.sign(IQc[lags]) * np.sign(QIc[lags]) * 1.0 < 0.0:
        # IQflip check:
        if phaseflip:
            logging.debug('confirmed: phaseflip between I and Q (90+ Deg ?)')
        else:
            logging.debug('phaseflip=False, IQflip=True')
    else:
        if phaseflip:
            logging.debug('phaseflip=True, IQflip=False')
        else:
            logging.debug('confirmed: no phaseflip')
    if phasefix:
        if nbasis:
            # Overwrite phase for photon number projection
            IIc = np.abs(psi)/2.0
            QQc = np.abs(psi)/2.0
            IQc = np.abs(psi)*0.0
            QIc = np.abs(psi)*0.0
        else:
            # Overwrite phase for squeezing projection
            IIc = np.abs(psi)/2.0
            QQc = -np.abs(psi)/2.0
            IQc = np.abs(psi)*0.0
            QIc = np.abs(psi)*0.0
    return np.array([IIc, IQc, QIc, QQc])


def lowPass(data, lowpass_sigma=0.0):
    if bool(lowpass_sigma):
        data = gaussian_filter1d(data, lowpass_sigma)  # Gausfilter
    return data


def makeheader_hyb(td, dicData):
    res = dicData['res']
    mapdim = td['mapdim']
    res.IQmapsON = np.zeros([4, mapdim[0], mapdim[1]])  # (4xON + 4xOFF) x 2D-Histogramms
    res.IQmapsOFF = np.zeros([4, mapdim[0], mapdim[1]])  # (4xON + 4xOFF) x 2D-Histogramms
    res.IQXbins = np.zeros([4, mapdim[0]+1])  # Bins are defined (from-to) therefore + 1
    res.IQYbins = np.zeros([4, mapdim[1]+1])  # Note : X and Y bins are shared between ON and OFF
    res.IQmapsON[0], res.IQXbins[0], res.IQYbins[0] = np.histogram2d(res.on_IQ[1, 0], res.on_IQ[1, 2], mapdim)
    res.IQmapsON[1], res.IQXbins[1], res.IQYbins[1] = np.histogram2d(res.on_IQ[1, 0], res.on_IQ[1, 3], mapdim)
    res.IQmapsON[2], res.IQXbins[2], res.IQYbins[2] = np.histogram2d(res.on_IQ[1, 1], res.on_IQ[1, 2], mapdim)
    res.IQmapsON[3], res.IQXbins[3], res.IQYbins[3] = np.histogram2d(res.on_IQ[1, 1], res.on_IQ[1, 3], mapdim)
    res.IQmapsOFF[0], res.IQXbins[0], res.IQYbins[0] = np.histogram2d(res.off_IQ[0, 0], res.off_IQ[0, 2], mapdim)
    res.IQmapsOFF[1], res.IQXbins[1], res.IQYbins[1] = np.histogram2d(res.off_IQ[0, 0], res.off_IQ[0, 3], mapdim)
    res.IQmapsOFF[2], res.IQXbins[2], res.IQYbins[2] = np.histogram2d(res.off_IQ[0, 1], res.off_IQ[0, 2], mapdim)
    res.IQmapsOFF[3], res.IQXbins[3], res.IQYbins[3] = np.histogram2d(res.off_IQ[0, 1], res.off_IQ[0, 3], mapdim)
    header_string_dim3 = ',' + td['dim1 name'] + ',' +str(td['dim1 start']) + ',' +str(td['dim1 stop'])
    res.headerII = ('Units,Counts,I1,' + str(res.IQXbins[0][0]) + ',' + str(res.IQXbins[0][-2]) + ',I2,' + str(res.IQYbins[0][0]) + ',' + str(res.IQYbins[0][-2]) + header_string_dim3)
    res.headerQQ = ('Units,Counts,Q1,' + str(res.IQXbins[1][0]) + ',' + str(res.IQXbins[1][-2]) + ',Q2,' + str(res.IQYbins[1][0]) + ',' + str(res.IQYbins[1][-2]) + header_string_dim3)
    res.headerIQ = ('Units,Counts,I1,' + str(res.IQXbins[2][0]) + ',' + str(res.IQXbins[2][-2]) + ',Q2,' + str(res.IQYbins[2][0]) + ',' + str(res.IQYbins[2][-2]) + header_string_dim3)
    res.headerQI = ('Units,Counts,Q1,' + str(res.IQXbins[3][0]) + ',' + str(res.IQXbins[3][-2]) + ',I2,' + str(res.IQYbins[3][0]) + ',' + str(res.IQYbins[3][-2]) + header_string_dim3)


def makehist2d_hyb(res, mapdim):
    res.IQmapsON = np.zeros([4, mapdim[0], mapdim[1]])  # (4xON + 4xOFF) x 2D-Histogramms
    res.IQmapsOFF = np.zeros([4, mapdim[0], mapdim[1]])  # (4xON + 4xOFF) x 2D-Histogramms
    res.IQXbins = np.zeros([4, mapdim[0]+1])  # Bins are defined (from-to) therefore + 1
    res.IQYbins = np.zeros([4, mapdim[1]+1])  # Note : X and Y bins are shared between ON and OFF
    # res.ON_IQ : on11, 12, 21, 22 # on I1I2 # on I1Q2 # on Q1I2 # on Q1Q2
    # res.OFF_IQ : [0:off12, 1:off21] each [0:I1, 1:Q1, 2:I2, 3:Q2] # off I1I2 # off I1Q2 # off Q1I2 # off Q1Q2
    res.IQmapsON[0], res.IQXbins[0], res.IQYbins[0] = np.histogram2d(res.on_IQ[1, 0], res.on_IQ[1, 2], [res.IQXbins[0], res.IQYbins[0]])
    res.IQmapsON[1], res.IQXbins[1], res.IQYbins[1] = np.histogram2d(res.on_IQ[1, 0], res.on_IQ[1, 3], [res.IQXbins[1], res.IQYbins[1]])
    res.IQmapsON[2], res.IQXbins[2], res.IQYbins[2] = np.histogram2d(res.on_IQ[1, 1], res.on_IQ[1, 2], [res.IQXbins[2], res.IQYbins[2]])
    res.IQmapsON[3], res.IQXbins[3], res.IQYbins[3] = np.histogram2d(res.on_IQ[1, 1], res.on_IQ[1, 3], [res.IQXbins[3], res.IQYbins[3]])
    res.IQmapsOFF[0], res.IQXbins[0], res.IQYbins[0] = np.histogram2d(res.off_IQ[0, 0], res.off_IQ[0, 2], [res.IQXbins[0], res.IQYbins[0]])
    res.IQmapsOFF[1], res.IQXbins[1], res.IQYbins[1] = np.histogram2d(res.off_IQ[0, 0], res.off_IQ[0, 3], [res.IQXbins[1], res.IQYbins[1]])
    res.IQmapsOFF[2], res.IQXbins[2], res.IQYbins[2] = np.histogram2d(res.off_IQ[0, 1], res.off_IQ[0, 2], [res.IQXbins[2], res.IQYbins[2]])
    res.IQmapsOFF[3], res.IQXbins[3], res.IQYbins[3] = np.histogram2d(res.off_IQ[0, 1], res.off_IQ[0, 3], [res.IQXbins[3], res.IQYbins[3]])
    res.IQmapsDiff = res.IQmapsON - res.IQmapsOFF


def get_covariance_submatrix(IQdata, lags, q):
    logging.debug('Calculating Submatrix')
    I1 = np.asarray(IQdata[0])
    Q1 = np.asarray(IQdata[1])
    I2 = np.asarray(IQdata[2])
    Q2 = np.asarray(IQdata[3])
    lags = int(lags)
    start = len(I1) - lags - 1
    stop = len(I1) + lags
    sub_matrix = np.zeros([4, lags * 2 + 1])
    sI1 = np.array(I1.shape)
    sQ2 = np.array(Q2.shape)
    shape = sI1 + sQ2 - 1
    fshape = [_next_regular(int(d)) for d in shape]  # padding to optimal size for FFTPACK
    fslice = tuple([slice(0, int(sz)) for sz in shape])  # remove padding later
    fftI1 = rfftn(I1, fshape)
    fftQ1 = rfftn(Q1, fshape)
    rfftI2 = rfftn(I2[::-1], fshape)
    rfftQ2 = rfftn(Q2[::-1], fshape)
    sub_matrix[0] = (irfftn((fftI1 * rfftI2))[fslice].copy()[start:stop] / len(fftI1))  # <II>
    sub_matrix[1] = (irfftn((fftI1 * rfftQ2))[fslice].copy()[start:stop] / len(fftI1))  # <IQ>
    sub_matrix[2] = (irfftn((fftQ1 * rfftI2))[fslice].copy()[start:stop] / len(fftI1))  # <QI>
    sub_matrix[3] = (irfftn((fftQ1 * rfftQ2))[fslice].copy()[start:stop] / len(fftI1))  # <QQ>
    q.put(sub_matrix)


def get_covariance_submatrix_full(IQdata, lags):
    I1 = np.asarray(IQdata[0])
    # Q1 = np.asarray(IQdata[1])
    # I2 = np.asarray(IQdata[2])
    Q2 = np.asarray(IQdata[3])
    lags = int(lags)
    start = len(I1) - lags - 1
    stop = len(I1) + lags
    sub_matrix = np.zeros([16, lags * 2 + 1])
    sI1 = np.array(I1.shape)
    sQ2 = np.array(Q2.shape)
    shape = sI1 + sQ2 - 1
    fshape = [_next_regular(int(d)) for d in shape]  # padding to optimal size for FFTPACK
    fslice = tuple([slice(0, int(sz)) for sz in shape])  # remove padding later
    fftIQ = 4*[None]
    rfftIQ = 4*[None]
    for i in range(4):
        fftIQ[i] = rfftn(IQdata[i], fshape)
        rfftIQ[i] = rfftn(IQdata[i][::-1], fshape)
    for j in range(4):
        for i in range(4):
            idx = i + j*4
            sub_matrix[idx] = (irfftn(fftIQ[i]*rfftIQ[j]))[fslice].copy()[start:stop]/len(fftIQ[i])
    return sub_matrix


if __name__ == '__main__':
    # testing individual functions
    pass
