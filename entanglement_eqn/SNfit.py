# -*- coding: utf-8 -*-
'''
@author: Ben Schneider
A script is used to readout mtx measurement data which also contains
a shotnoise responses.
Then fits them for G and Tn
'''
import numpy as np
from parsers import savemtx, loadmtx, make_header
# from scipy.optimize import curve_fit  # , leastsq
from scipy.constants import Boltzmann as Kb
from scipy.constants import h, e, c  # , pi
from scipy.ndimage.filters import gaussian_filter1d
from lmfit import minimize, Parameters, report_fit  # , Parameter
from matplotlib.pyplot import plot, hold, figure, show, title, ion, close


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
        (self.I1I1, d3,
         self.d2, self.d1, self.dz) = loadmtx(self.fifolder + self.filein1)
        self.Q1Q1, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein2)
        self.I2I2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein3)
        self.Q2Q2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein4)
        self.Vm, self.d3, dv2, dv1, dvz = loadmtx(self.fifolder + self.filein5)
        self.lags0 = find_nearest(d1.lin, 0.0)  # lags position
        self.Ib0 = find_nearest(d3.lin, 0.0)  # Zero current position

    def loadCcor(self):
        '''
        want to simply load the amplitude at max correlation position
        i.e. at lags = 0
        '''
        self.I1I2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein6)
        self.I1Q2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein7)
        self.Q1I2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein8)
        self.Q1Q2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein9)
        self.I1Q1, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein10)
        self.I2Q2, d3, d2, d1, dz = loadmtx(self.fifolder + self.filein11)
        self.cPD1 = (self.I1I1[self.lags0]+self.Q1Q1[self.lags0])
        self.cPD2 = (self.I2I2[self.lags0]+self.Q2Q2[self.lags0])

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
        self.dIV[self.dIV > 100.0] = 50.0  # Replace any value larger than 200
        self.dIVlp = gaussian_filter1d(abs(self.dIV), self.LP)  # Gausfilter

    def make_cvals(self, cpt=5, snr=3):
        '''
        Using this function to obtain the amount of noise present
        in the background while ignoring the regions where the cross corr...
        are expected (5pt around the zero lags position).

        this function calculates the uncertainty values in the given cross corr

        denotation for dict key values beginning with:
            r* for raw data directly extracted
            u* for uncertainty values calculated
            n* for normalized values (considering uncertainty worst values)
        what is does:
        1. create dict keys raw data values at lags0
        2. create zero arrays in dictionary
        3. fill those zero arrays with corresponding uncertainty values
        4. calculate norm values and store them in the dict
        '''
        r1i = self.cvals['rI1I1'] = self.I1I1[self.lags0, :, :]
        r1q = self.cvals['rQ1Q1'] = self.Q1Q1[self.lags0, :, :]
        r2i = self.cvals['rI2I2'] = self.I2I2[self.lags0, :, :]
        r2q = self.cvals['rQ2Q2'] = self.Q2Q2[self.lags0, :, :]
        self.cvals['rI1I2'] = self.I1I2[self.lags0, :, :]
        self.cvals['rI1Q2'] = self.I1Q2[self.lags0, :, :]
        self.cvals['rQ1I2'] = self.Q1I2[self.lags0, :, :]
        self.cvals['rQ1Q2'] = self.Q1Q2[self.lags0, :, :]
        self.cvals['uI1I1'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['uQ1Q1'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['uI2I2'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['uQ2Q2'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['uI1I2'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['uI1Q2'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['uQ1I2'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['uQ1Q2'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['nI1I1'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['nQ1Q1'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['nI2I2'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['nQ2Q2'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['nI1I2'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['nI1Q2'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['nQ1I2'] = np.zeros([self.d2.pt, self.d3.pt])
        self.cvals['nQ1Q2'] = np.zeros([self.d2.pt, self.d3.pt])

        for x2 in range(self.d2.pt):
            for x3 in range(self.d3.pt):
                # fill uncertainty values
                u1i = self.calU(self.I1I1[:, x2, x3], self.lags0, cpt)
                u1q = self.calU(self.Q1Q1[:, x2, x3], self.lags0, cpt)
                u2i = self.calU(self.I2I2[:, x2, x3], self.lags0, cpt)
                u2q = self.calU(self.Q2Q2[:, x2, x3], self.lags0, cpt)
                uii = self.calU(self.I1I2[:, x2, x3], self.lags0, cpt)
                uiq = self.calU(self.I1Q2[:, x2, x3], self.lags0, cpt)
                uqi = self.calU(self.Q1I2[:, x2, x3], self.lags0, cpt)
                uqq = self.calU(self.Q1Q2[:, x2, x3], self.lags0, cpt)
                rii = self.cvals['rI1I2'][x2, x3]
                riq = self.cvals['rI1Q2'][x2, x3]
                rqi = self.cvals['rQ1I2'][x2, x3]
                rqq = self.cvals['rQ1Q2'][x2, x3]
                self.cvals['uI1I1'][x2, x3] = u1i
                self.cvals['uQ1Q1'][x2, x3] = u1q
                self.cvals['uI2I2'][x2, x3] = u2i
                self.cvals['uQ2Q2'][x2, x3] = u2q
                self.cvals['uI1I2'][x2, x3] = uii
                self.cvals['uI1Q2'][x2, x3] = uiq
                self.cvals['uQ1I2'][x2, x3] = uqi
                self.cvals['uQ1Q2'][x2, x3] = uqq
                # calculate the normed values and store them in the matrix
                self.cvals['nI1I1'][x2, x3] = r1i[x2, x3]  # +u1i/2
                self.cvals['nQ1Q1'][x2, x3] = r1q[x2, x3]  # +u1q/2
                self.cvals['nI2I2'][x2, x3] = r2i[x2, x3]  # +u2i/2
                self.cvals['nQ2Q2'][x2, x3] = r2q[x2, x3]  # +u2q/2
                # that error is already added from the shot noise values
                self.cvals['nI1I2'][x2, x3] = rii
                # +uii/2 if rii < 0 else rii-uii/2
                self.cvals['nI1Q2'][x2, x3] = riq
                # +uiq/2 if riq < 0 else riq-uiq/2
                self.cvals['nQ1I2'][x2, x3] = rqi
                # +uqi/2 if rqi < 0 else rqi-uqi/2
                self.cvals['nQ1Q2'][x2, x3] = rqq
                # +uqq/2 if rqq < 0 else rqq-uqq/2
                # 0 if the requested SnR ratio is not met
                if abs(rii) < snr*uii/2.0:
                    self.cvals['nI1I2'][x2, x3] = 0.0
                if abs(riq) < snr*uiq/2.0:
                    self.cvals['nI1Q2'][x2, x3] = 0.0
                if abs(rqi) < snr*uqi/2.0:
                    self.cvals['nQ1I2'][x2, x3] = 0.0
                if abs(rqq) < snr*uqq/2.0:
                    self.cvals['nQ1Q2'][x2, x3] = 0.0

    def calU(self, z1, lags0, cpt):
        '''
        This function removes from an array
        cpt points around the lag0 position
        and returns the square root variance of this new array.

        Get background noise value of the cross correlation data.
        '''
        z2 = z1[:lags0-cpt]*1.0
        z3 = z1[lags0+cpt:]*1.0
        return abs(np.sqrt(np.var(np.concatenate([z2, z3]))))


'''
                    2 Classes END
Beginning of the functions used for fitting and such
-------------------------------------------------------
'''


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
    R = vc.dRm
    E1 = (e*vc.I*R+h*vc.f)/(2*Kb*T)
    E2 = (e*vc.I*R-h*vc.f)/(2*Kb*T)
    Si = ((2*Kb*T/R) * (E1/np.tanh(E1) + E2/np.tanh(E2)))
    return (vc.B*G*(Si * R**2 +
                    4.0*Kb*T*R + 4.0*Kb*Tn*vc.Z0 *
                    (vc.Z0**2+vc.Zopt*vc.Zopt)/(vc.Z0*vc.Z0+vc.Zopt*vc.Zopt)) *
            (vc.Z0/((R+vc.Z0)*(vc.dRm+vc.Z0))))


def fitfun2(params, vc, digi):
    '''
    req: params with G, Tn, T; and vc as variable carrier
    return: fitting value or array
    also set digi='D1' or digi='D2'
    '''
    T = params['T'].value
    if digi == 'D1':
        G = params['G1'].value
        Tn = params['Tn1'].value
        # T = params['T1'].value
        vc.f = vc.f1
    if digi == 'D2':
        G = params['G2'].value
        Tn = params['Tn2'].value
        # T = params['T2'].value
        vc.f = vc.f2
    return fitfunc(G, Tn, T, vc)


def ministuff(params, vc, measd, digi='D1'):
    '''
    req: params with G, Tn, T
         I (current array or value) (Amps)
         dRm (Resistance for this current)
         digi='D1' or 'D2'
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
    SNfit = fitfun2(params, vc, digi)
    SNfit[vc.crop[0][0]:vc.crop[0][1]] = 0
    measd[vc.crop[0][0]:vc.crop[0][1]] = 0
    SNfit[0:(vc.crop[1][0])] = 0
    SNfit[vc.crop[1][1]:-1] = 0
    measd[0:(vc.crop[1][0])] = 0
    measd[vc.crop[1][1]:-1] = 0
    return (measd-SNfit)*1e10


def bigstuff(params, vc, pidx):
    '''
    return all 3 as one combined thing (shotnoise 1,2 and photon differences)
    '''
    data1 = vc.cPD1[pidx]*1.0
    data2 = vc.cPD2[pidx]*1.0
    snd1 = ministuff(params, vc, data1, digi='D1')
    snd2 = ministuff(params, vc, data2, digi='D2')
    ph1, ph2 = photoncal(params, pidx, vc)
    phtd = abs(1-ph1/ph2) if abs(1-ph1/ph2) > 0.15 else 1.0
    zpfmin = 1-ph1-ph2 if 1-ph1-ph2 > 0.0 else 0.0  # zpf: min 0.5 photons
    return abs(snd1)*abs(snd2)*phtd + zpfmin*1e9
    # return phtd


def photoncal(params, pidx, vc):
    '''
    calculates the difference in photon numbers at the hemt input
    for a current bias of Ib0 (0 uA) and after the amplifier photons and temp
    photons are subtracted.
    '''
    d1 = vc.cPD1[pidx]*1.0
    d2 = vc.cPD2[pidx]*1.0
    p1 = d1[vc.Ib0]/(h*vc.f1*vc.B*params['G1'].value)
    p2 = d2[vc.Ib0]/(h*vc.f2*vc.B*params['G2'].value)
    Amp1 = Kb*params['Tn1'].value/(h*vc.f1)
    Amp2 = Kb*params['Tn2'].value/(h*vc.f2)
    Phot1 = p1 - Amp1  # -Ent1
    Phot2 = p2 - Amp2  # -Ent2
    phT1 = blackbodyrad(vc.f1, params['T'].value)  # est phot num from Temp
    phT2 = blackbodyrad(vc.f2, params['T'].value)
    # Need to make sure that the number of photons are strictly more than 0.5
    # since that is due to the zpf! thus Phot1+Phot2 - 1 > 0
    return Phot1-phT1, Phot2-phT2
    # return abs(1-Phot1/Phot2) if abs(1-Phot1/Phot2) > 0.15 else 1.0


def blackbodyrad(f, T):
    '''
    returns the number of photons expected for a certain temperature present
    Energy of photons(f,T): (2*h*f**3)/(c**2)*1/(np.e(h*f/(Kb*T))-1)
    (see Plank's law) units of Power/angle/meter^2/unit Bandwidth
    4*pi/c
    '''
    spectral_radiance = 2*f*f/(c*c*(np.e**(h*f/(Kb*T))-1))
    area = 7*4e-4  # 7x4 cm^2 chip
    return spectral_radiance*area
    # spectral_phot_density = spectral_radiance*4*np.pi/c
    # return spectral_phot_density


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
    if plotFit is True:
        close('all')
        ion()
    SNr = SN_class()
    vc.load_and_go()
    vc.calc_diff_resistance()
    # create crop vector for the fitting
    crop_within = find_nearest(vc.I, -0.4e-6), find_nearest(vc.I, 0.5e-6)
    crop_outside = find_nearest(vc.I, -19e-6), find_nearest(vc.I, 19e-6)
    vc.crop = [crop_within, crop_outside]
    # create fitting parameters
    params = Parameters()
    params.add('Tn1', value=3.7, vary=True, min=2.0, max=4.0)
    params.add('G1', value=3.38e7, vary=True, min=1e6, max=1e9)
    params.add('Tn2', value=3.7, vary=True, min=2.0, max=4.0)
    params.add('G2', value=3.38e7, vary=True, min=1e6, max=1e9)
    # params.add('T1', value=0.05, vary=False, min=0.01, max=0.25)
    # params.add('T2', value=0.05, vary=False, min=0.01, max=0.25)
    params.add('T', value=vc.Texp, vary=False, min=0.007, max=0.38)
    data1 = vc.cPD1*1.0
    data2 = vc.cPD2*1.0
    for pidx in range(vc.cPD1.shape[0]):
        '''
        scales Voltage_trace[selected power] to Volts
        obtains differential Resistance Rm
        fits selected data set
        records corresponding fit results into SN_r class values
        '''
        vc.dRm = vc.dIVlp[pidx]  # select dRm
        result = minimize(ministuff, params, args=(vc, data1[pidx]*1.0, 'D1'))
        result = minimize(ministuff,
                          result.params, args=(vc, data2[pidx]*1.0, 'D2'))

        # now fit all of them together:
        # result.params['T1'].vary = True
        # result.params['T2'].vary = True
        result.params['T'].vary = True
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

    # print 'Photons in1', SNr.Pi1.mean(), '+/-', SNr.Pi1del.mean()
    # print 'Photons in2', SNr.Pi2.mean(), '+/-', SNr.Pi2del.mean()
    # print 'Pn1', SNr.Pn1.mean(), '+/-', SNr.Pn1del.mean()
    # print 'Pn2', SNr.Pn2.mean(), '+/-', SNr.Pn2del.mean()

    return SNr


def plotSNfit(result, vc, pidx, digi='D1'):
    ''' result : fitting results
        vc, variable carrier
        pidx power index
        digi = 'D1' or 'D2'
        '''
    if digi == 'D1':
        data = vc.cPD1[pidx]*1.0
        SNfit = fitfun2(result.params, vc, 'D1')
        Pn = (result.params['G1'].value*vc.B *
              (Kb*(result.params['Tn1'].value)+0.5*h*vc.f1))
    if digi == 'D2':
        data = vc.cPD2[pidx]*1.0
        SNfit = fitfun2(result.params, vc, 'D2')
        Pn = (result.params['G2'].value*vc.B *
              (Kb*(result.params['Tn2'].value)+0.5*h*vc.f2))

    Pnarray = np.ones(len(vc.I))*Pn
    figure()
    title2 = (digi + ', RF-Drive: ' + str(vc.d2.lin[pidx]))
    plot(vc.I, data*1e9)
    hold(True)
    plot(vc.I, SNfit*1e9)
    plot(vc.I, Pnarray*1e9)
    title(title2)
    hold(False)
    show()
