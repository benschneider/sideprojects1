# -*- coding: utf-8 -*-
'''
@author: Ben Schneider
A script is used to readout mtx measurement data which also contains
a shotnoise responses.
Then fits them for G and Tn
'''
import numpy as np
from parsers import savemtx, loadmtx, make_header, read_header
# from scipy.optimize import curve_fit  # , leastsq
from scipy.constants import Boltzmann as kB
from scipy.constants import h, e, c  # , pi
from scipy.ndimage.filters import gaussian_filter1d
from lmfit import minimize, Parameters, report_fit  # , Parameter
import PyGnuplot as gp

'''
The Two classes SN_class used to store the fit results
------------------------------------------------------
and the variable carrier used to carry and deal with variables that are
passed over quite often
'''


class SN_class():
    '''
    This is simply an empty class which i am going to use
    to store the shot-noise fitting results
    '''

    def __init__(self):
        ''' Using empty lists at which i can append incoming data'''
        self.G1del = []
        self.G2del = []
        self.Tn1del = []
        self.Tn2del = []
        self.G1 = []
        self.G2 = []
        self.Tn1 = []
        self.Tn2 = []
        self.T1 = []
        self.T2 = []
        self.T = []
        self.Tdel = []


class variable_carrier():
    ''' used to store and pass lots of variables and locations
        simply create with vc = my_variable_class()
        it currently has the following default settings in __init__:
        self.Z0 = 50.0
        self.Zopt = 50.0
        self.B = 1e5
        self.f1 = 4.1e9
        self.f2 = 4.8e9
        self.RTR = 1009.1 * 1e3           # Ib Resistance in Ohm
        self.RG = 1000.0                  # Pre Amp gain factor
        self.filein1 = 'S1_949_G0mV_SN_PCovMat_cI1I1.mtx'
        self.filein2 = 'S1_949_G0mV_SN_PCovMat_cQ1Q1.mtx'
        self.filein3 = 'S1_949_G0mV_SN_PCovMat_cI2I2.mtx'
        self.filein4 = 'S1_949_G0mV_SN_PCovMat_cQ2Q2.mtx'
        self.filein5 = 'S1_949_G0mV_SN_PV'
        self.fifolder = 'sn_data//'
        Cross corr files
        self.filein6 = 'S1_949_G0mV_SN_PCovMat_cI1I2.mtx'
        self.filein7 = 'S1_949_G0mV_SN_PCovMat_cI1Q2.mtx'
        self.filein8 = 'S1_949_G0mV_SN_PCovMat_cQ1I2.mtx'
        self.filein9 = 'S1_949_G0mV_SN_PCovMat_cQ1Q2.mtx'
        self.fifolder = 'sn_data//'
    '''

    def __init__(self):
        self.LP = 3                 # Gaus-Filter i.e. Low-Pass Vm derivative
        self.Z0 = 50.0
        self.Zopt = 50.0
        self.B = 1e5
        self.f1 = 4.1e9
        self.f2 = 4.8e9
        self.RTR = 1009.1 * 1e3           # Ib Resistance in Ohm
        self.RG = 1000.0                  # Pre Amp gain factor
        self.cvals = {}         # Cross corr dictionary with key value elements

    def load_and_go(self):
        '''
        simply executes the sub definitions
        loads data,
        normalizes to SI units,
        calculates differential resistances
        '''
        self.loaddata()
        self.loadCcor()
        self.norm_to_SI()

    def loaddata(self):
        '''
        Loads the data defined in self.filein1 ..
        This loads the shotnoise relevant data files
        '''
        self.I1I1, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein1, True)
        self.Q1Q1, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein2, True)
        self.I2I2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein3, True)
        self.Q2Q2, self.d3I, self.d2, self.d1, self.dz = loadmtx(self.fifolder + self.filein4, True)
        self.Vm, self.d3, dv2, dv1, dvz = loadmtx(self.fifolder + self.filein5, True)
        self.lags0 = find_nearest(self.d1.lin, 0.0)  # lags position
        self.Ib0 = find_nearest(self.d3.lin, 0.0)  # Zero current position

    def loadCcor(self):
        '''
        want to simply load the amplitude at max correlation position
        i.e. at lags = 0
        '''
        self.I1I2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein6, True)
        self.I1Q2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein7, True)
        self.Q1I2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein8, True)
        self.Q1Q2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein9, True)
        self.I1Q1, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein10, True)
        self.I2Q2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein11, True)
        # fix single pixel shifts in the data.
        self.I1I2 = self.f1pN(self.I1I2)
        self.I1Q2 = self.f1pN(self.I1Q2)
        self.Q1I2 = self.f1pN(self.Q1I2)
        self.Q1Q2 = self.f1pN(self.Q1Q2)
        # self.PD1 = (self.I1I1[self.lags0] + self.Q1Q1[self.lags0])
        # self.PD2 = (self.I2I2[self.lags0] + self.Q2Q2[self.lags0])
        # self.cPD1 = ( np.abs((self.I1I1[self.lags0]) + np.abs(self.Q1Q1[self.lags0])) +
        #               (np.abs(self.I2I2[self.lags0]) + np.abs(self.Q2Q2[self.lags0])) )

        # self.psi( ((self.I1I2[self.lags0]) - (self.Q1Q2[self.lags0])) +
        #              1j * ((self.Q1I2[self.lags0]) + (self.I1Q2[self.lags0])))
        # self.phase0 = np.angle(self.psy)
        # self.mag0 =
        self.cPD1 = ((abs(self.I1I1[self.lags0]) + abs(self.Q1Q1[self.lags0])) +
                     (abs(self.I2I2[self.lags0]) + abs(self.Q2Q2[self.lags0])))
        self.cPD2 = ((abs(self.I1I2[self.lags0]) + abs(self.Q1Q2[self.lags0])) +
                     (abs(self.Q1I2[self.lags0]) + abs(self.I1Q2[self.lags0])))

    def f1pN(self, array3, d=2):
        '''
        d is the distance of points to search for the peak position around the lags0 pos.
        Insert for example <I1I2> array data where the center peak is not at pos lags0
        it returns a new <I1I2> array slightly rolled such that the center peak is at pos lags0.
        '''
        for i in range(array3.shape[2]):
            for j in range(array3.shape[1]):
                tArray = array3[:, j, i]*1.0  # copy temp work array
                distance = (np.argmax(np.abs(tArray[self.lags0-d:self.lags0+d+1])) - d)*-1
                array3[:, j, i] = np.roll(tArray, distance)

        return array3

    def norm_to_SI(self):
        '''
        Take amplifier gains and resistances as defined by self.RTR and self.RG
        to scale voltage units to [Volt] and [Amps]
        '''
        self.d3.scale = 1.0 / (self.RTR)     # scale X-axis to Amps
        self.d3.update_lin()
        self.I = self.d3.lin
        self.Vm = self.Vm / self.RG          # scale Vm-data to Volts

    def calc_diff_resistance(self):
        '''
        calculates the differential resistance of all traces in the
        variable carrier
        '''
        self.d3step = self.d3.lin[1] - self.d3.lin[0]   # get step-size
        self.dIV = xderiv(self.Vm[0], self.d3step)
        if self.LP > 0.0:
            self.dIVlp = gaussian_filter1d(abs(self.dIV), self.LP)  # Gausfilter
        else:
            self.dIVlp = abs(self.dIV)
        # self.dIVlp[self.dIVlp > 100.0] = 0.0

    def make_cvals(self, cpt=5, snr=2):
        '''
        Using this function to obtain the amount of noise present
        in the background while ignoring the regions where the cross corr...
        are expected (5pt around the zero lags position).
        '''
        self.Sarr = np.zeros([8, self.d2.pt, self.d3.pt])
        self.Narr = np.zeros([8, self.d2.pt, self.d3.pt])
        self.Varr = np.zeros([8, self.d2.pt, self.d3.pt])
        S = np.zeros(8)
        N = np.zeros(8)
        V = np.zeros(8)
        for x2 in range(self.d2.pt):
            for x3 in range(self.d3.pt):
                S[0], N[0] = get_SNR(self.I1I1[:, x2, x3], cpt)
                S[1], N[1] = get_SNR(self.Q1Q1[:, x2, x3], cpt)
                S[2], N[2] = get_SNR(self.I2I2[:, x2, x3], cpt)
                S[3], N[3] = get_SNR(self.Q2Q2[:, x2, x3], cpt)
                V[:4] = S[:4]
                S[4], N[4] = get_SNR(self.I1I2[:, x2, x3], cpt)
                S[5], N[5] = get_SNR(self.I1Q2[:, x2, x3], cpt)
                S[6], N[6] = get_SNR(self.Q1I2[:, x2, x3], cpt)
                S[7], N[7] = get_SNR(self.Q1Q2[:, x2, x3], cpt)
                V[4] = S[4] if abs(S[4]) > snr * N[4] else 0.0
                V[5] = S[5] if abs(S[5]) > snr * N[5] else 0.0
                V[6] = S[6] if abs(S[6]) > snr * N[6] else 0.0
                V[7] = S[7] if abs(S[7]) > snr * N[7] else 0.0
                self.Sarr[:, x2, x3] = S
                self.Narr[:, x2, x3] = N
                self.Varr[:, x2, x3] = V


def get_SNR(array, distance):
    pos0 = find_absPeakPos(array, distance)
    offset = getOffset(array, pos0, 4)
    signal = array[pos0] - offset
    noise = calU(array, pos0, distance)
    return signal, noise


def calU(z1, lags0, cpt):
    '''
    This function removes from an array
    cpt points around the lag0 position
    and returns the square root variance of this new array.
    Get background noise value of the cross correlation data.
    '''
    z2 = z1[:lags0 - cpt] * 1.0
    z3 = z1[lags0 + cpt:] * 1.0
    return abs(np.sqrt(np.var(np.concatenate([z2, z3]))))/2.0


def getOffset(z1, lags0, cpt):
    '''
    This function removes from an array
    cpt points around the lag0 position
    and returns the mean offset
    '''
    z2 = z1[:lags0 - cpt] * 1.0
    z3 = z1[lags0 + cpt:] * 1.0
    return abs(np.mean(np.concatenate([z2, z3])))


def find_switch(array, threshold=1e-9):
    '''
    array is a 1-D array
    threshold is the limit of passable point to point changes.
    finds a position at which a sudden switch/jump is present on the order of
    switch
    A switch is found by two successive jumps, one down/up and then another.
    (of course this will give a false response for two steps in succession)
    If these happen in succession we found a single pixel error
    (returns only the first point it finds then it stops searching)
    '''
    a = array[0]
    idx = 0
    for mm, val in enumerate(array):
        if abs(a-val) > threshold:
            if (mm-1 == idx):  # if prev and current position had a jump we found one
                return idx
            idx = mm
        a = val


def find_absPeakPos(someArray, dist=1):
    '''
    finds within a short range around the center of
    an array the peak/ dip value position
    and returns value at the position.
    1. abs(someArray)
    2. crop someArray with the range
    3. max(someArray)
    assumes that the mean of someArray = 0
    '''
    Array = np.abs(someArray * 1.0)
    A0 = int(len(Array)/2)  # A0 center pos (round down)
    pos = np.argmax(Array[A0-dist:A0+dist+1])+A0-dist
    return pos


def xderiv(d2MAT, dx=1.0, axis=0):
    '''
    This  derivative is inaccurate at the edges.
    Calculates a 3p derivative of a 1D, 2D matrix.
    This does not require you to shift the xaxis by one half pt.
    dx = distance between points
    '''
    if len(d2MAT.shape) > 1:
        if axis == 1:
            ''' Not tested yet could be faster than a matrix transpose'''
            a2 = np.zeros([d2MAT.shape[0] + 2, d2MAT.shape[1]])
            a2[1:-1, :] = d2MAT
            m1 = d2MAT - a2[:-2, :]
            m2 = a2[2:, :] - d2MAT
            dy = (m1 + m2) / 2.0
            dy[0, :] = dy[1, :]
            dy[-1, :] = dy[-2, :]
        elif axis == 0:
            a2 = np.zeros([d2MAT.shape[0], d2MAT.shape[1] + 2])
            a2[:, 1:-1] = d2MAT
            m1 = d2MAT - a2[:, :-2]
            m2 = a2[:, 2:] - d2MAT
            dy = (m1 + m2) / 2.0
            dy[:, 0] = dy[:, 1]
            dy[:, -1] = dy[:, -2]
        return dy / dx
    else:
        a2 = np.zeros([d2MAT.shape[0] + 2])
        a2[1:-1] = d2MAT
        m1 = d2MAT - a2[:-2]
        m2 = a2[2:] - d2MAT
        dy = (m1 + m2) / 2.0
        dy[0] = dy[1]
        dy[-1] = dy[-2]
        return dy / dx


def find_nearest(someArray, value):
    ''' This function helps to find the index corresponding to a value
    in an array.
    Usage: indexZero = find_nearest(myarray, 0.0)
    returns: abs(myarray-value).argmin()
    '''
    idx = abs(someArray - value).argmin()
    return idx


def cropTrace(trace, vc):
    '''
    'crops' a trace with the specifics given by
    sets values to zero:
    with the range of
    vc.crop[0][0] to vc.crop[0][1]
    the outer edges
    fist index til vc.crop[1][0]
    vc.crop[1][1] til last index
    crop values for example create with:
    crop_within = find_nearest(I, -0.9e-6), find_nearest(I, 1.1e-6)
    crop_outside = find_nearest(I, -19.5e-6), find_nearest(I, 19.5e-6)
    crop = [crop_within, crop_outside]
    '''
    newarray = trace*1.0
    newarray[vc.crop[0][0]:vc.crop[0][1]] = 0.0
    newarray[0:(vc.crop[1][0])] = 0.0
    newarray[vc.crop[1][1]:-1] = 0.0
    return np.array(newarray)


def fitfunc(G, Tn, T, f, vc):
    '''
    This contains the new fitting equation, which i use to fit the
    shot noise response.
    returns: fit-value(x, ...)
    Amplifier and circulator impedance are both assumed to be Z0
    '''
    mf = (vc.Z0 / (vc.Z0 + vc.dRm))**2.0
    vvzpf = h * f * vc.Z0 / 2.0
    vvsn = 2.0 * e * np.abs(vc.I) * vc.dRm * vc.dRm * mf
    vvnt = 4.0 * kB * T * vc.dRm * mf + 1e-99
    # vvnt = (4.0 * kB * vc.Tz * vc.Z0 * mf) + (4.0 * kB * T * vc.dRm * mf) + 1e-99
    E1 = (vvsn + vvzpf) / (vvnt)
    E2 = (vvsn - vvzpf) / (vvnt)
    Svi = vvnt / (2.0 * vc.Z0) * (E1 / np.tanh(E1) + E2 / np.tanh(E2))
    return vc.B * G * (Svi + kB * Tn)


def get_residuals(params, vc, pidx, digi='D1'):
    '''
    returns residuals, and difference between min(data) - min(fit)
    '''
    T = params['T'].value
    vc.Tz = params['Tz'].value

    if digi == 'D1':
        G1 = params['G1'].value
        Tn1 = params['Tn1'].value
        factor = vc.f1*h*vc.B*G1  # factor to photon #
        data = np.array(vc.cPD1[pidx]) * 1.0
        SNf = fitfunc(G1, Tn1, T, vc.f1, vc)

    if digi == 'D2':
        G2 = params['G2'].value
        Tn2 = params['Tn2'].value
        factor = vc.f2*h*vc.B*G2
        data = np.array(vc.cPD2[pidx]) * 1.0
        SNf = fitfunc(G2, Tn2, T, vc.f2, vc)

    res = np.array(np.abs((data - SNf)/factor))
    res2 = res  # cropTrace(res, vc)
    # pmin = np.abs(data.min() - SNf.min())/factor  # adding additional weight to respect min values
    # scpos = vc.Ib0
    # p = np.abs(data[scpos-1:scpos+15] - SNf[scpos-1:scpos+15])/factor
    # res2[scpos-1:scpos+15] = p

    d0 = np.mean(np.sort(data)[:5])/factor
    d1 = np.mean(np.sort(SNf)[:5])/factor
    p2 = np.abs(d0-d1)

    return (1 + res2)*(1 + p2) - 1


def bigstuff(params, vc, pidx):
    '''
    return all 3 as one combined thing (shotnoise 1,2 and photon differences)
    '''
    res1 = get_residuals(params, vc, pidx, digi='D1')
    res2 = get_residuals(params, vc, pidx, digi='D2')
    # return abs(res1 + 1) * abs(res2 + 1) - 1.0
    return abs(res1) + abs(res2)


def DoSNfits(vc, plotFit=False):
    '''
    Loading the data files I1I1, Q1Q1, I2I2, Q2Q2, Vm
    d1, d2, d3 are all the same since they all originate from the same type of
    measurement.
    before running it needs details of files and parameters to use.
    Those are made by creating a variables_class;
    example:
    vc = my_variables_class()
    which contains the following default settings
    vc.Z0 = 50.0
    vc.Zopt = 50.0
    vc.B = 1e5
    vc.f1 = 4.1e9
    vc.f2 = 4.8e9
    vc.RTR = 1009.1 * 1e3           # Ib Resistance in Ohms
    vc.RG = 1000.0                  # Pre Amp gain factor
    additionally the filenames need to be defined in there:
    simply give the base filenames as:
    vc.filein1 = 'S1_949_G0mV_SN_PCovMat_cI1I1.mtx'
    vc.filein2 = 'S1_949_G0mV_SN_PCovMat_cQ1Q1.mtx'
    vc.filein3 = 'S1_949_G0mV_SN_PCovMat_cI2I2.mtx'
    vc.filein4 = 'S1_949_G0mV_SN_PCovMat_cQ2Q2.mtx'
    vc.filein5 = 'S1_949_G0mV_SN_PV.mtx'
    and of course the folder where to find these files
    vc.fifolder = 'sn_data//'
    Right now this def getSNfits does too many things for a single definition:
        - loads the defined mtx files into the vc class
    '''
    # if plotFit is True:
    #     plt.close('all')
    #     # plt.ion()
    SNr = SN_class()
    vc.load_and_go()
    vc.calc_diff_resistance()

    # create fitting parameters
    params = Parameters()
    params.add('Tn1', value=8.46, vary=False, min=0.0, max=25.0)
    params.add('G1', value=7.70e7, vary=True, min=1e3, max=1e17)
    params.add('Tn2', value=14.7, vary=True, min=0.0, max=25.0)
    params.add('G2', value=1.6e7, vary=True, min=1e3, max=1e17)
    params.add('T', value=vc.Texp, vary=False, min=0.0001, max=0.1)
    params.add('Tz', value=0.0, vary=False, min=0.000, max=0.050)

    for pidx in range(vc.cPD1.shape[0]):
        '''
        scales Voltage_trace[selected power] to Volts
        obtains differential Resistance Rm
        fits selected data set
        records corresponding fit results into SN_r class values
        '''
        # vc.dRm = vc.dIVlp[pidx, ::-1]  # select dRm
        vc.dRm = vc.dIVlp[pidx]  # select dRm
        if vc.Ravg > 0.0:
            vc.dRm = np.ones_like(vc.dRm)*vc.Ravg
        # correct diff/Resistance at SC branch:
        vc.dRm[vc.Ib0] = 0.0
        result = minimize(get_residuals, params, args=(vc, pidx, 'D1'))
        # result.params['T'].vary = True
        # result.params['G2'].value = result.params['G1'].value
        result = minimize(get_residuals, result.params, args=(vc, pidx, 'D2'))

        # now fit all of them together:
        result = minimize(bigstuff, result.params, args=(vc, pidx))
        print 'RF power', vc.d2.lin[pidx]
        print report_fit(result)
        SNr.G1del.append(result.params['G1'].stderr)
        SNr.Tn1del.append(result.params['Tn1'].stderr)
        SNr.G1.append(result.params['G1'].value)
        SNr.Tn1.append(result.params['Tn1'].value)
        SNr.G2del.append(result.params['G2'].stderr)
        SNr.Tn2del.append(result.params['Tn2'].stderr)
        SNr.G2.append(result.params['G2'].value)
        SNr.Tn2.append(result.params['Tn2'].value)
        SNr.T.append(result.params['T'].value)
        SNr.Tdel.append(result.params['T'].stderr)
        if plotFit is True:
            plotSNfit(result, vc, pidx, 'D1')
        if plotFit is True:
            plotSNfit(result, vc, pidx, 'D2')

    # lists to array
    SNr.G1 = np.array(SNr.G1)
    SNr.G2 = np.array(SNr.G2)
    SNr.Tn1 = np.array(SNr.Tn1)
    SNr.Tn2 = np.array(SNr.Tn2)
    SNr.G1del = np.array(SNr.G1del)
    SNr.G2del = np.array(SNr.G2del)
    SNr.Tn1del = np.array(SNr.Tn1del)
    SNr.Tn2del = np.array(SNr.Tn2del)

    # Photon numbers hemt input
    SNr.Pi1 = (kB * SNr.Tn1) / (h * vc.f1) + 0.5
    SNr.Pi1del = (kB * SNr.Tn1del) / (h * vc.f1)
    SNr.Pi2 = (kB * SNr.Tn2) / (h * vc.f2) + 0.5
    SNr.Pi2del = (kB * SNr.Tn2del) / (h * vc.f2)

    # Noise power at output at I = 0
    SNr.Pn1 = SNr.G1 * vc.B * SNr.Pi1 * (h * vc.f1)
    SNr.Pn1del = (SNr.Pn1 * np.sqrt((SNr.G1del / SNr.G1)**2 +
                                    (SNr.Tn1del / SNr.Tn1)**2))
    SNr.Pn2 = SNr.G2 * vc.B * SNr.Pi2 * (h * vc.f2)
    SNr.Pn2del = (SNr.Pn2 * np.sqrt((SNr.G2del / SNr.G2)**2 +
                                    (SNr.Tn2del / SNr.Tn2)**2))
    return SNr


def plotSNfit(result, vc, pidx, digi='D1'):
    ''' result : fitting results
        vc, variable carrier
        pidx power index
        digi = 'D1' or 'D2'
        '''
    T = result.params['T'].value
    vc.Tz = result.params['Tz'].value

    if digi == 'D1':
        f = vc.f1
        G = result.params['G1'].value
        Tn = result.params['Tn1'].value
        factor = f*h*vc.B*G  # factor to photon #
        data = np.array(vc.cPD1[pidx]) * 1.0

    if digi == 'D2':
        f = vc.f2
        G = result.params['G2'].value
        Tn = result.params['Tn2'].value
        factor = f*h*vc.B*G
        data = np.array(vc.cPD2[pidx]) * 1.0

    SNf = fitfunc(G, Tn, T, f, vc)
    Amp = G*vc.B*kB*Tn

    title2 = (digi + ', RF-Drive: ' + str(vc.d2.lin[pidx]))
    dataname = digi + '_' + str(vc.d2.lin[pidx]) + '.dat'
    gp.figure()
    gp.c('clear')
    gp.s([vc.I, (data)/factor, (SNf)/factor, np.ones_like(data)*Amp/factor], filename=dataname)
    gp.c('set title "' + title2 + '"')
    gp.c('plot "'+dataname+'" u 1:2 w l t "Data" ')
    gp.c('replot "'+dataname+'" u 1:3 w l t "Fit" ')
    gp.c('replot "'+dataname+'" u 1:4 w l t "Amplifier Noise" ')
    gp.c('save "'+dataname[:-3]+'gn"')
