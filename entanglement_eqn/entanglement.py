# -*- coding: utf-8 -*-
'''
@author: Ben Schneider
Extract degree of entanglement from the data files
'''
import numpy as np
from scipy.constants import h, e
import SNfit as sn
from parsers import savemtx, make_header


def get_LogNegNum(CovM):
    '''
    function to calculate Log-Negativity from a covariance matrix

    CovM looks a bit like:
        I1  I1I1 I1Q1 I1I2 I1Q2
        Q1  Q1I1 Q1Q1 Q1I2 Q1Q2
        I2  I2I1 I2Q1 I2I2 I2Q2
        Q2  Q2I1 Q2Q1 Q2I2 Q2Q2

            I1   Q1   I2   Q2
    '''
    V = np.linalg.det(CovM)
    A = np.linalg.det(CovM[:2, :2])
    B = np.linalg.det(CovM[2:, 2:])
    C = np.linalg.det(CovM[:2, 2:])
    s = A + B - 2.0*C
    vn = np.sqrt(s/2.0 - np.sqrt(s**2.0 - 4.0*V) / 2)
    return -np.log10(2.0*vn)


def TwoModeSqueeze_inequality(f1, f2, n):
    ''' This function calculates the amount of two mode
    squeezing required to prove hard entanglement
    '''
    return ((2.0*np.sqrt(f1*f2)*(2*n)) / (f1*(2*n+1)+f2*(2*n+1)))


def createCovMat(vc, snd, power1=0, Ibx=0):
    ''' creates a covariance matrix
        for a selected power,
        (and Current bias currently the zero position)


    covM = np.array([[I1I1, I1Q1, I1I2, I1Q2],
                    [I1Q1, Q1Q1, Q1I2, Q1Q2],
                    [I1I2, Q1I2, I2I2, I2Q2],
                    [I1Q2, Q1Q2, I2Q2, Q2Q2]])
    return covM

    # Ibx = sn.find_nearest(vc.d3.lin, Ib)
    '''

    I1I1 = vc.I1I1[vc.lags0, power1, Ibx]
    Q1Q1 = vc.Q1Q1[vc.lags0, power1, Ibx]
    I2I2 = vc.I2I2[vc.lags0, power1, Ibx]
    Q2Q2 = vc.Q2Q2[vc.lags0, power1, Ibx]
    I1I2 = vc.I1I2[vc.lags0, power1, Ibx]
    I1Q2 = vc.I1Q2[vc.lags0, power1, Ibx]
    Q1I2 = vc.Q1I2[vc.lags0, power1, Ibx]
    Q1Q2 = vc.Q1Q2[vc.lags0, power1, Ibx]
    I1Q1 = vc.I1Q1[vc.lags0, power1, Ibx]
    I2Q2 = vc.I2Q2[vc.lags0, power1, Ibx]

    I1Q1 = 0.0
    I2Q2 = 0.0
    # To convert to photon input numbers at the Hemt input
    g1 = snd.G1[power1]*h*vc.f1*vc.B
    g2 = snd.G2[power1]*h*vc.f2*vc.B
    g12 = np.sqrt(g1*g2)

    # Added Noise Photons by the Amp (without zpf)
    a1 = (snd.Pi1[power1] - 0.5)/2.0
    a2 = (snd.Pi2[power1] - 0.5)/2.0

    covM = np.array([[I1I1/g1-a1, I1Q1/g1, I1I2/g12, I1Q2/g12],
                    [I1Q1/g1, Q1Q1/g1-a1, Q1I2/g12, Q1Q2/g12],
                    [I1I2/g12, Q1I2/g12, I2I2/g2-a2, I2Q2/g2],
                    [I1Q2/g12, Q1Q2/g12, I2Q2/g2, Q2Q2/g2-a2]])

    # if power1 < 0.0001:
    #     ''' If the drive power is zero the powers are known to be 0.25! '''
    #     covM = np.array([[0.25, I1Q1/g1, I1I2/g12, I1Q2/g12],
    #                     [I1Q1/g1, 0.25, Q1I2/g12, Q1Q2/g12],
    #                     [I1I2/g12, Q1I2/g12, 0.25, I2Q2/g2],
    #                     [I1Q2/g12, Q1Q2/g12, I2Q2/g2, 0.25]])

    return covM


def NMatrix(vc, snd):
    '''
    This assembles the Log neg matrix LnM
    '''
    LnM = np.zeros([1, vc.d2.pt, vc.d3.pt])
    for ii in range(vc.d2.pt):
        for jj in range(vc.d3.pt):
            CovM = createCovMat(vc, snd, ii, jj)
            N = get_LogNegNum(CovM)
            # if N < 0:
            #     N = 0
            LnM[0, ii, jj] = N

    return LnM


vc = sn.variable_carrier()
savename = '957_G27mV_LogN.mtx'
vc.filein1 = 'S1_957_G27mV_SNCovMat_cI1I1.mtx'
vc.filein2 = 'S1_957_G27mV_SNCovMat_cQ1Q1.mtx'
vc.filein3 = 'S1_957_G27mV_SNCovMat_cI2I2.mtx'
vc.filein4 = 'S1_957_G27mV_SNCovMat_cQ2Q2.mtx'
vc.filein5 = 'S1_957_G27mV_SNV.mtx'
vc.filein6 = 'S1_957_G27mV_SNCovMat_cI1I2.mtx'
vc.filein7 = 'S1_957_G27mV_SNCovMat_cI1Q2.mtx'
vc.filein8 = 'S1_957_G27mV_SNCovMat_cQ1I2.mtx'
vc.filein9 = 'S1_957_G27mV_SNCovMat_cQ1Q2.mtx'
vc.filein10= 'S1_957_G27mV_SNCovMat_cI1Q1.mtx'
vc.filein11= 'S1_957_G27mV_SNCovMat_cI2Q2.mtx'
# vc.filein11 = vc.filein10
vc.fifolder = 'sn_data//'
vc.LP = 2.2
vc.load_and_go()
snd = sn.DoSNfits(vc)

LnM = NMatrix(vc, snd)

headtxt = make_header(vc.d3, vc.d2, vc.d1, meas_data='Log-Negativity')
savemtx(savename, LnM, headtxt)
