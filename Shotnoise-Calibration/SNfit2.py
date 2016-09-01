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
import sys, os

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
        self.T1del = []
        self.T2 = []
        self.T2del = []
        self.G12del = []
        self.G12cdel = []
        self.Tn12del = []
        self.Tn12cdel = []
        self.G12 = []
        self.G12c = []
        self.Tn12 = []
        self.Tn12c = []
        self.T12 = []
        self.T12del = []
        self.T12c = []
        self.T12cdel = []


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
        self.f1 = 4.8e9
        self.f2 = 4.1e9
        self.RTR = 1009.1 * 1e3           # Ib Resistance in Ohm
        self.RG = 1000.0                  # Pre Amp gain factor
        self.cvals = {}         # Cross corr dictionary with key value elements
        self.resultfolder = 'result_folder//'
        self.inclMin = 15
        self.Tn1 = 8.46
        self.Tn2 = 6.0
        self.G1 = 7.67e7
        self.G2 = 7.6e7
        self.T = 0.007
        self.snr = 1.0
        self.cpt = 4.0

    def load_and_go(self, gpFolder=False):
        '''
        # gpfolder is a switch to make gnuplot change folder if required
        simply executes the sub definitions
        loads data,
        normalizes to SI units,
        calculates differential resistances
        '''
        self.loaddata()
        self.loadCcor()
        self.norm_to_SI()
        self.make_cvals()
        if not os.path.exists(self.resultfolder):
            os.makedirs(self.resultfolder)

        if gpFolder:
            gp.c('cd "' + self.resultfolder + '"')

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
        self.cPD1 = (self.I1I1[self.lags0] + self.Q1Q1[self.lags0])
        self.cPD2 = (self.I2I2[self.lags0] + self.Q2Q2[self.lags0])
        self.cPD3 = ((abs(self.I1I1[self.lags0]) + abs(self.Q1Q1[self.lags0])) +
                     (abs(self.I2I2[self.lags0]) + abs(self.Q2Q2[self.lags0])))
        self.cPD4 = ((abs(self.I1I2[self.lags0]) + abs(self.Q1Q2[self.lags0])) +
                     (abs(self.Q1I2[self.lags0]) + abs(self.I1Q2[self.lags0])))

    def f1pN(self, array3, d=1):
        '''
        d is the distance of points to search for the peak position around the lags0 pos.
        Insert for example <I1I2> array data where the center peak is not at pos lags0
        it returns a new <I1I2> array slightly rolled such that the center peak is at pos lags0.
        '''
        for i in range(array3.shape[2]):
            for j in range(array3.shape[1]):
                tArray = array3[:, j, i]*1.0  # copy temp work array
                # Only roll the data if the signal to Noise ratio is larger than 2
                if np.max(np.abs(tArray[self.lags0-d:self.lags0+d+1])) > 2.0*np.var(tArray):
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

    def make_cvals(self):
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
                S[0], N[0] = get_SNR(self.I1I1[:, x2, x3], self.cpt)
                S[1], N[1] = get_SNR(self.Q1Q1[:, x2, x3], self.cpt)
                S[2], N[2] = get_SNR(self.I2I2[:, x2, x3], self.cpt)
                S[3], N[3] = get_SNR(self.Q2Q2[:, x2, x3], self.cpt)
                V[:4] = S[:4]
                S[4], N[4] = get_SNR(self.I1I2[:, x2, x3], self.cpt)
                S[5], N[5] = get_SNR(self.I1Q2[:, x2, x3], self.cpt)
                S[6], N[6] = get_SNR(self.Q1I2[:, x2, x3], self.cpt)
                S[7], N[7] = get_SNR(self.Q1Q2[:, x2, x3], self.cpt)
                # Max values
                # V[4] = np.sign(S[4]) * (abs(S[4]) + abs(N[4])) if (abs(S[4]) - self.snr*abs(N[4])) > 0.0 else 0.0
                # V[5] = np.sign(S[5]) * (abs(S[5]) + abs(N[5])) if (abs(S[5]) - self.snr*abs(N[5])) > 0.0 else 0.0
                # V[6] = np.sign(S[6]) * (abs(S[6]) + abs(N[6])) if (abs(S[6]) - self.snr*abs(N[6])) > 0.0 else 0.0
                # V[7] = np.sign(S[7]) * (abs(S[7]) + abs(N[7])) if (abs(S[7]) - self.snr*abs(N[7])) > 0.0 else 0.0
                # Min Values
                # V[4] = np.sign(S[4]) * (abs(S[4]) - abs(N[4])) if (abs(S[4]) - self.snr*abs(N[4])) > 0.0 else 0.0
                # V[5] = np.sign(S[5]) * (abs(S[5]) - abs(N[5])) if (abs(S[5]) - self.snr*abs(N[5])) > 0.0 else 0.0
                # V[6] = np.sign(S[6]) * (abs(S[6]) - abs(N[6])) if (abs(S[6]) - self.snr*abs(N[6])) > 0.0 else 0.0
                # V[7] = np.sign(S[7]) * (abs(S[7]) - abs(N[7])) if (abs(S[7]) - self.snr*abs(N[7])) > 0.0 else 0.0
                # Med Values
                V[4] = S[4] if (abs(S[4]) - self.snr*abs(N[4])) > 0.0 else 0.0
                V[5] = S[5] if (abs(S[5]) - self.snr*abs(N[5])) > 0.0 else 0.0
                V[6] = S[6] if (abs(S[6]) - self.snr*abs(N[6])) > 0.0 else 0.0
                V[7] = S[7] if (abs(S[7]) - self.snr*abs(N[7])) > 0.0 else 0.0
                self.Sarr[:, x2, x3] = S
                self.Narr[:, x2, x3] = N
                self.Varr[:, x2, x3] = V


def get_SNR(array, distance):
    pos0 = int(find_absPeakPos(array, distance))
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
    lags0 = int(lags0)
    cpt = int(cpt)
    z2 = z1[:lags0 - cpt] * 1.0
    z3 = z1[lags0 + cpt:] * 1.0
    return abs(np.sqrt(np.var(np.concatenate([z2, z3]))))


def getOffset(z1, lags0, cpt):
    '''
    This function removes from an array
    cpt points around the lag0 position
    and returns the mean offset
    '''
    z2 = z1[:lags0 - cpt] * 1.0
    z3 = z1[lags0 + cpt:] * 1.0
    return abs(np.mean(np.concatenate([z2, z3])))


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
    dist = int(dist)
    Array = np.abs(someArray * 1.0)
    A0 = int(len(Array)/2)  # A0 center pos (round down)
    pos = np.argmax(Array[A0-dist:A0+dist+1])+A0-dist
    return pos


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


def get_residuals(params, vc, pidx, mtype='D1'):
    '''
    returns residuals, and difference between min(data) - min(fit)
    '''
    vc.Tz = params['Tz'].value

    if mtype == 'D1':
        T1 = params['T1'].value
        G1 = params['G1'].value
        Tn1 = params['Tn1'].value
        factor = vc.f1*h*vc.B*G1  # factor to photon #
        data = np.array(vc.cPD1[pidx]) * 1.0
        SNf = fitfunc(G1, Tn1, T1, vc.f1, vc)

    if mtype == 'D2':
        T2 = params['T2'].value
        G2 = params['G2'].value
        Tn2 = params['Tn2'].value
        factor = vc.f2*h*vc.B*G2
        data = np.array(vc.cPD2[pidx]) * 1.0
        SNf = fitfunc(G2, Tn2, T2, vc.f2, vc)

    if mtype == 'D12':
        T12 = params['T12'].value
        G12 = params['G12'].value
        Tn12 = params['Tn12'].value
        factor = vc.f1*h*vc.B*G12  # factor to photon #
        data = np.array(vc.cPD3[pidx]) * 1.0
        SNf = fitfunc(G12, Tn12, T12, vc.f1, vc)

    if mtype == 'D12c':
        T12c = params['T12c'].value
        G12c = params['G12c'].value
        Tn12c = params['Tn12c'].value
        factor = vc.f1*h*vc.B*G12c  # factor to photon #
        data = np.array(vc.cPD4[pidx]) * 1.0
        SNf = fitfunc(G12c, Tn12c, T12c, vc.f1, vc)

    res = np.array(np.abs((data - SNf)/factor))
    res2 = cropTrace(res, vc)
    # pmin = np.abs(data.min() - SNf.min())/factor  # adding additional weight to respect min values
    # scpos = vc.Ib0
    # p = np.abs(data[scpos-1:scpos+13] - SNf[scpos-1:scpos+13])/factor
    # res2[scpos-1:scpos+13] = p

    p2 = 0.0
    if vc.inclMin > 0:
        d0 = np.mean(np.sort(data)[:vc.inclMin])/factor
        d1 = np.mean(np.sort(SNf)[:vc.inclMin])/factor
        p2 = np.abs(d0-d1)  # obtain the differences of the lowest values

    return res2 * (1.0 + p2 * 1.0)


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
    params1 = Parameters()
    params2 = Parameters()
    params3 = Parameters()
    params4 = Parameters()

    # Why this brainfart mess?? Cross fitting later... :/

    params1.add('Tz', value=0.0, vary=False, min=0.0, max=0.090)
    params2.add('Tz', value=0.0, vary=False, min=0.0, max=0.090)
    params3.add('Tz', value=0.0, vary=False, min=0.0, max=0.090)
    params4.add('Tz', value=0.0, vary=False, min=0.0, max=0.090)

    params1.add('T1', value=vc.T, vary=False, min=0.0001, max=0.1)
    params1.add('Tn1', value=vc.Tn1, vary=True, min=0.0, max=25.0)
    params1.add('G1', value=vc.G1, vary=True, min=1e3, max=1e17)

    params2.add('T2', value=vc.T, vary=False, min=0.0, max=0.1)
    params2.add('Tn2', value=vc.Tn2, vary=True, min=0.0, max=25.0)
    params2.add('G2', value=vc.G2, vary=True, min=1e3, max=1e17)

    params3.add('T12', value=vc.T, vary=False, min=0.0, max=0.1)
    params3.add('Tn12', value=8.0, vary=True, min=0.0, max=25.0)
    params3.add('G12', value=8.6e7, vary=True, min=1e3, max=1e17)

    params4.add('T12c', value=vc.T, vary=False, min=0.0, max=0.1)
    params4.add('Tn12c', value=5.0, vary=True, min=0.0, max=25.0)
    params4.add('G12c', value=1.6e7, vary=True, min=1e3, max=1e17)

    for pidx in range(vc.cPD1.shape[0]):
        '''
        scales Voltage_trace[selected power] to Volts
        obtains differential Resistance Rm
        fits selected data set
        records corresponding fit results into SN_r class values
        '''
        vc.dRm = vc.dIVlp[pidx]  # select dRm
        if vc.Ravg > 0.0:
            vc.dRm = np.ones_like(vc.dRm)*vc.Ravg

        # vc.dRm[vc.Ib0] = 0.0  # correct diff/Resistance at SC branch:

        result1 = minimize(get_residuals, params1, args=(vc, pidx, 'D1'))
        result2 = minimize(get_residuals, params2, args=(vc, pidx, 'D2'))
        result3 = minimize(get_residuals, params3, args=(vc, pidx, 'D12'))
        result4 = minimize(get_residuals, params4, args=(vc, pidx, 'D12c'))
        print report_fit(result1)
        print report_fit(result2)
        print report_fit(result3)
        print report_fit(result4)

        # now fit all of them together:
        # result = minimize(bigstuff, result.params, args=(vc, pidx))
        print 'RF power', vc.d2.lin[pidx]

        SNr.T1.append(result1.params['T1'].value)
        SNr.T1del.append(result1.params['T1'].stderr)
        SNr.G1del.append(result1.params['G1'].stderr)
        SNr.Tn1del.append(result1.params['Tn1'].stderr)
        SNr.G1.append(result1.params['G1'].value)
        SNr.Tn1.append(result1.params['Tn1'].value)

        SNr.T2.append(result2.params['T2'].value)
        SNr.T2del.append(result2.params['T2'].stderr)
        SNr.G2del.append(result2.params['G2'].stderr)
        SNr.Tn2del.append(result2.params['Tn2'].stderr)
        SNr.G2.append(result2.params['G2'].value)
        SNr.Tn2.append(result2.params['Tn2'].value)

        SNr.T12.append(result3.params['T12'].value)
        SNr.T12del.append(result3.params['T12'].stderr)
        SNr.G12del.append(result3.params['G12'].stderr)
        SNr.Tn12del.append(result3.params['Tn12'].stderr)
        SNr.G12.append(result3.params['G12'].value)
        SNr.Tn12.append(result3.params['Tn12'].value)

        SNr.T12c.append(result4.params['T12c'].value)
        SNr.T12cdel.append(result4.params['T12c'].stderr)
        SNr.G12cdel.append(result4.params['G12c'].stderr)
        SNr.Tn12cdel.append(result4.params['Tn12c'].stderr)
        SNr.G12c.append(result4.params['G12c'].value)
        SNr.Tn12c.append(result4.params['Tn12c'].value)

        if plotFit is True:
            plotSNfit(result1, vc, pidx, 'D1')
            plotSNfit(result2, vc, pidx, 'D2')
            plotSNfit(result3, vc, pidx, 'D12')
            plotSNfit(result4, vc, pidx, 'D12c')

    # lists to array
    SNr.G1 = np.array(SNr.G1)
    SNr.G1del = np.array(SNr.G1del)
    SNr.G2 = np.array(SNr.G2)
    SNr.G2del = np.array(SNr.G2del)
    SNr.G12 = np.array(SNr.G12)
    SNr.G12del = np.array(SNr.G12del)
    SNr.G12c = np.array(SNr.G12c)
    SNr.G12cdel = np.array(SNr.G12cdel)
    SNr.Tn1 = np.array(SNr.Tn1)
    SNr.Tn1del = np.array(SNr.Tn1del)
    SNr.Tn2 = np.array(SNr.Tn2)
    SNr.Tn2del = np.array(SNr.Tn2del)
    SNr.Tn12 = np.array(SNr.Tn12)
    SNr.Tn12del = np.array(SNr.Tn12del)
    SNr.Tn12c = np.array(SNr.Tn12c)
    SNr.Tn12cdel = np.array(SNr.Tn12cdel)

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

    with open(vc.resultfolder + 'results.txt', 'w+') as output:
        for variable in dir(SNr):
            value = getattr(SNr, variable)
            output.write(variable+':'+str(value)+'\n')

    return SNr, result1, result2, result3, result4


def plotSNfit(result, vc, pidx, digi='D1'):
    ''' result : fitting results
        vc, variable carrier
        pidx power index
        digi = 'D1' or 'D2'
        '''
    vc.Tz = result.params['Tz'].value

    if digi == 'D1':
        f = vc.f1
        T = result.params['T1'].value
        G = result.params['G1'].value
        Tn = result.params['Tn1'].value
        data = np.array(vc.cPD1[pidx]) * 1.0

    if digi == 'D2':
        f = vc.f2
        T = result.params['T2'].value
        G = result.params['G2'].value
        Tn = result.params['Tn2'].value
        data = np.array(vc.cPD2[pidx]) * 1.0

    if digi == 'D12':
        f = vc.f1
        T = result.params['T12'].value
        G = result.params['G12'].value
        Tn = result.params['Tn12'].value
        data = np.array(vc.cPD3[pidx]) * 1.0

    if digi == 'D12c':
        f = vc.f1
        T = result.params['T12c'].value
        G = result.params['G12c'].value
        Tn = result.params['Tn12c'].value
        data = np.array(vc.cPD4[pidx]) * 1.0

    factor = f*h*vc.B*G  # factor to photon #
    SNf = fitfunc(G, Tn, T, f, vc)
    Amp = G*vc.B*kB*Tn

    title2 = (digi + ', RF-Drive ' + str(vc.d2.lin[pidx]) + ' G ' + str(np.round(G/1e7, 2)) +
              'e7 = ' + str(np.round(np.log10(G)*10.0, 2)) + 'dB  T ' + str(np.round(Tn, 2)) + 'K')
    dataname = digi + '_' + str(vc.d2.lin[pidx]) + '.dat'
    gp.c('')
    gp.figure()
    # gp.c('clear')
    gp.s([vc.I*1e6, (data)/factor, (SNf)/factor, np.ones_like(data)*Amp/factor],
         filename=vc.resultfolder+dataname)
    gp.c('set title "' + title2 + '"')
    gp.c('set xrange[-19:19]')
    gp.c('set key center top')
    gp.c('plot "' + dataname + '" u 1:2 w l t "Data" ')
    gp.c('replot "' + dataname + '" u 1:3 w l t "Fit" ')
    gp.c('replot "' + dataname + '" u 1:4 w l t "Amplifier Noise" ')
    gp.c('save "' + dataname[:-3] + 'gn"')
    gp.pdf(dataname[:-3]+'pdf')
    print dataname[:-3]+'pdf'
