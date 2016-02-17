# -*- coding: utf-8 -*-
'''
@author: Ben Schneider
A script is used to readout mtx measurement data which also contains
a shotnoise responses.
Then fits them for G and Tn
'''

import numpy as np
from parsers import savemtx, loadmtx, make_header
from scipy.optimize import curve_fit  # , leastsq
# import scipy.optimize
from scipy.constants import Boltzmann as Kb
from scipy.constants import h, e  # , pi
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from lmfit import minimize, Parameters, report_fit  # , Parameter


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
            a2 = np.zeros([d2MAT.shape[0]+2, d2MAT.shape[1]])
            a2[1:-1, :] = d2MAT
            m1 = d2MAT - a2[:-2, :]
            m2 = a2[2:, :] - d2MAT
            dy = (m1+m2)/2.0
            dy[0, :] = dy[1, :]
            dy[-1, :] = dy[-2, :]
        elif axis == 0:
            a2 = np.zeros([d2MAT.shape[0], d2MAT.shape[1]+2])
            a2[:, 1:-1] = d2MAT
            m1 = d2MAT - a2[:, :-2]
            m2 = a2[:, 2:] - d2MAT
            dy = (m1+m2)/2.0
            dy[:, 0] = dy[:, 1]
            dy[:, -1] = dy[:, -2]
        return dy/dx
    else:
        a2 = np.zeros([d2MAT.shape[0]+2])
        a2[1:-1] = d2MAT
        m1 = d2MAT - a2[:-2]
        m2 = a2[2:] - d2MAT
        dy = (m1+m2)/2.0
        dy[0] = dy[1]
        dy[-1] = dy[-2]
        return dy/dx


def find_nearest(someArray, value):
    ''' This function helps to find the index corresponding to a value
    in an array.
    Usage: indexZero = find_nearest(myarray, 0.0)
    returns: abs(myarray-value).argmin()
    '''
    idx = abs(someArray-value).argmin()
    return idx


def fitfunc(G, Tn, T, vc):
    # def fitfunc(x, G, Tn, T, c):
    '''
    This contains the fitting equation, which i use to fit the
    shot noise response.
    returns: fit-value(x, ...)
    '''
    E1 = (e*vc.I*vc.dRm+h*vc.f)/(2*Kb*T)
    E2 = (e*vc.I*vc.dRm-h*vc.f)/(2*Kb*T)
    Si = ((2*Kb*T/vc.dRm) * (E1/np.tanh(E1) + E2/np.tanh(E2)))
    return (vc.B*G*(Si * vc.dRm**2 +
                    4.0*Kb*T*vc.dRm + 4.0*Kb*Tn*vc.Z0 *
                    (vc.dRm**2+vc.Zopt*vc.Zopt)/(vc.Z0*vc.Z0+vc.Zopt*vc.Zopt)) *
            (vc.Z0/((vc.dRm+vc.Z0)*(vc.dRm+vc.Z0))))


def fitfun2(params, vc):
    '''
    req: params with G, Tn, T; and vc as variable carrier
    return: fitting value or array
    '''
    G = params['G'].value
    Tn = params['Tn'].value
    T = params['T'].value
    return fitfunc(G, Tn, T, vc)


def ministuff(params, vc, measd):
    '''
    req: params with G, Tn, T
         I (current array or value) (Amps)
         dRm (Resistance for this current)
         measured data (value or array)
         crop values for example create with:
         crop_within = find_nearest(I, -0.9e-6), find_nearest(I, 1.1e-6)
         crop_outside = find_nearest(I, -19.5e-6), find_nearest(I, 19.5e-6)
         crop = [crop_within, crop_outside]
         This crop is used to cut data corresponding to the current values
         i.e. to cut away the critical current part (from to (crop within))
         also edges where the derivative and filter is inaccurate (crop outside)
    returns: residuals*1e10;
         (difference between measured and fitted data after it has been croped)
    '''
    SNfit = fitfun2(params, vc)
    SNfit[vc.crop[0][0]:vc.crop[0][1]] = 0
    measd[vc.crop[0][0]:vc.crop[0][1]] = 0
    SNfit[0:(vc.crop[1][0])] = 0
    SNfit[vc.crop[1][1]:-1] = 0
    measd[0:(vc.crop[1][0])] = 0
    measd[vc.crop[1][1]:-1] = 0
    return (measd-SNfit)*1e10


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


class my_variables_class():
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
        self.filein1 = 'S1_949_G0mV_SN_PCovMat_cI1I1.mtx'
        self.filein2 = 'S1_949_G0mV_SN_PCovMat_cQ1Q1.mtx'
        self.filein3 = 'S1_949_G0mV_SN_PCovMat_cI2I2.mtx'
        self.filein4 = 'S1_949_G0mV_SN_PCovMat_cQ2Q2.mtx'
        self.filein5 = 'S1_949_G0mV_SN_PV.mtx'
        self.fifolder = 'sn_data//'

    def load_and_go(self):
        '''
        simply executes the sub definitions
        loads data, normalizes to SI units, calculates differential resistances
        '''
        self.loaddata()
        self.norm_to_SI()
        self.calc_diff_resistance()

    def loaddata(self):
        '''
        Loads the data defined in self.filein1 .. 5
        '''
        (self.I1I1, d3,
         self.d2, self.d1, self.dz) = loadmtx(self.fifolder + self.filein1)
        self.Q1Q1, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein2)
        self.I2I2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein3)
        self.Q2Q2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein4)
        self.Vm, self.d3, dv2, dv1, dvz = loadmtx(self.fifolder + self.filein5)

    def norm_to_SI(self):
        '''
        Take amplifier gains and resistances as defined by self.RTR and self.RG
        to scale voltage units to [Volt] and [Amps]
        '''
        self.d3.scale = 1.0/(self.RTR)     # scale X-axis to Amps
        self.d3.update_lin()
        self.I = self.d3.lin
        self.Vm = self.Vm/self.RG          # scale Vm-data to Volts

    def calc_diff_resistance(self):
        '''
        calculates the differential resistance of all traces in one go
        '''
        self.d3step = self.d3.lin[1] - self.d3.lin[0]   # get step-size
        self.dIV = xderiv(self.Vm[0], self.d3step)
        self.dIVlp = gaussian_filter1d(abs(self.dIV), self.LP)  # Gausfilter


def getSNfits(vc):
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
        -
    '''
    SNr = SN_class()
    vc.load_and_go()

    lags0 = find_nearest(vc.d1.lin, 0.0)  # lags position
    PD1 = (vc.I1I1[lags0]+vc.Q1Q1[lags0])
    PD2 = (vc.I2I2[lags0]+vc.Q2Q2[lags0])

    # create crop vector for the fitting
    crop_within = find_nearest(vc.I, -1.55e-6), find_nearest(vc.I, 1.55e-6)
    crop_outside = find_nearest(vc.I, -19e-6), find_nearest(vc.I, 19e-6)
    vc.crop = [crop_within, crop_outside]

    # create fitting parameters
    params = Parameters()
    params.add('Tn', value=3.7, vary=True)  # , min=2.2, max=5)
    params.add('G', value=3.38e7, vary=True)  # , min=1e6, max=1e9)
    params.add('T', value=0.012, vary=False, min=0.01, max=0.5)

    data1 = PD1*1.0
    data2 = PD2*1.0
    for pidx in range(PD1.shape[0]):
        '''
        scales Voltage_trace[selected power] to Volts
        obtains differential Resistance Rm
        fits selected data set
        records corresponding fit results into SN_r class values
        '''
        vc.dRm = vc.dIVlp[pidx]    # select dRm which is wanted

        vc.f = vc.f1
        result = minimize(ministuff, params, args=(vc, data1[pidx]*1.0))
        print report_fit(result)
        SNr.G1del.append(result.params['G'].stderr)
        SNr.Tn1del.append(result.params['Tn'].stderr)
        SNr.G1.append(result.params['G'].value)
        SNr.Tn1.append(result.params['Tn'].value)
        SNfit1 = fitfun2(result.params, vc)

        Pn1 = (result.params['G'].value*vc.B *
               (Kb*(result.params['Tn'].value+result.params['T'])+0.5*h*vc.f1))
        Pn1array = np.ones(len(vc.I))*Pn1

        vc.f = vc.f2
        result = minimize(ministuff, params, args=(vc, data2[pidx]*1.0))
        print report_fit(result)
        SNr.G2del.append(result.params['G'].stderr)
        SNr.Tn2del.append(result.params['Tn'].stderr)
        SNr.G2.append(result.params['G'].value)
        SNr.Tn2.append(result.params['Tn'].value)
        SNfit2 = fitfun2(result.params, vc)

        Pn2 = (result.params['G'].value*vc.B *
               (Kb*(result.params['Tn'].value+result.params['T'])+0.5*h*vc.f2))
        Pn2array = np.ones(len(vc.I))*Pn2

        vc.f = vc.f1
        plt.figure()
        title = ('D1, RF-Drive: ' + str(vc.d2.lin[pidx]))
        plt.plot(vc.I, data1[pidx]*1e9)
        plt.hold(True)
        plt.plot(vc.I, SNfit1*1e9)
        plt.plot(vc.I, Pn1array*1e9)
        plt.title(title)
        plt.hold(False)
        plt.show()

        vc.f = vc.f2
        plt.figure()
        title = ('D2, RF-Drive: ' + str(vc.d2.lin[pidx]))
        plt.plot(vc.I, data2[pidx]*1e9)
        plt.hold(True)
        plt.plot(vc.I, SNfit2*1e9)
        plt.plot(vc.I, Pn2array*1e9)
        plt.title(title)
        plt.hold(False)
        plt.show()

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
    SNr.Pi1 = (Kb*SNr.Tn1)/(h*vc.f1) + 0.5
    SNr.Pi1del = (Kb*SNr.Tn1del)/(h*vc.f1)
    SNr.Pi2 = (Kb*SNr.Tn2)/(h*vc.f2) + 0.5
    SNr.Pi2del = (Kb*SNr.Tn2del)/(h*vc.f2)

    # Noise power at output at I = 0
    SNr.Pn1 = SNr.G1 * vc.B * SNr.Pi1 * (h * vc.f1)
    SNr.Pn1del = (SNr.Pn1 * np.sqrt((SNr.G1del/SNr.G1)**2 +
                                    (SNr.Tn1del/SNr.Tn1)**2))
    SNr.Pn2 = SNr.G2 * vc.B * SNr.Pi2 * (h * vc.f2)
    SNr.Pn2del = (SNr.Pn2 * np.sqrt((SNr.G2del/SNr.G2)**2 +
                                    (SNr.Tn2del/SNr.Tn2)**2))

    print 'Photons in1', SNr.Pi1.mean(), '+/-', SNr.Pi1del.mean()
    print 'Photons in2', SNr.Pi2.mean(), '+/-', SNr.Pi2del.mean()
    print 'Pn1', SNr.Pn1.mean(), '+/-', SNr.Pn1del.mean()
    print 'Pn2', SNr.Pn2.mean(), '+/-', SNr.Pn2del.mean()

    return SNr
