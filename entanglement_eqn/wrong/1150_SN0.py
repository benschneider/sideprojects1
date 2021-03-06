# -*- coding: utf-8 -*-
'''
@author: Ben Schneider
Extract degree of entanglement from the data files
'''
import numpy as np
from scipy.constants import h, e
from scipy.constants import Boltzmann as kB
import SNfit2 as sn
from parsers import savemtx, make_header
# import math


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
    sigma = A + B - 2.0 * C

    if sigma*sigma-4.0*V < 0.0:
        return 0.0

    vn = np.sqrt(sigma / 2.0 - np.sqrt(sigma * sigma - 4.0 * V) / 2.0)

    # if C == 0 : vn = vn*8   # N should be strictly negative for no Corr
    # return -np.log10(2.0*vn)  # maybe this should be 4*vn
    if C == 0:
        return 0.0
    else:
        return -np.log10(2.0 * vn) if (np.log10(2.0 * vn)) < 0.0 else 0.0


def get_sqIneq(vc, CovM):
    '''
    This calculates the amount of two mode squeezing
    and then subtracts the amount required for proving that it is entangled
    Such that a positive number corresponds to a breach in
    the inequality equation and thus can only be explained by a highly
    entangled signal.
    This is a strong indicator for the presence of entanglement
    '''
    # Photon numbers numbers at f1 and f2:
    n1 = CovM[0, 0] * 1.0 + CovM[1, 1] * 1.0
    n2 = CovM[2, 2] * 1.0 + CovM[3, 3] * 1.0
    # Photons detected to be TMS:
    sqp1 = CovM[0, 2] * 1.0 - CovM[1, 3] * 1.0 + 1.0 * 1j * (CovM[0, 3] + CovM[1, 2])
    squeezing = np.abs(sqp1) / ((n1 + n2) / 2.0)
    # inequality equation:
    ineq = ((2.0 * np.sqrt(vc.f1 * vc.f2) + (n1 + n2)) /
            (vc.f1 * (2.0 * n1 + 1.0) + vc.f2 * (2.0 * n2 + 1.0)))
    if (squeezing - ineq) > 0.0:
        return squeezing
        # if 1e-6 < get_LogNegNum(CovM):  # check if LogNeg is Breached
        #     return squeezing
        # else:
        #     return squeezing*-1
    else:
        return 0.0


def TwoModeSqueeze_inequality(f1, f2, n):
    ''' This function calculates the amount of two mode
    squeezing required to prove hard entanglement
    '''
    return ((2.0 * np.sqrt(f1 * f2) * (2 * n)) / (f1 * (2 * n + 1) + f2 * (2 * n + 1)))


def createCovMat(vc, power1=0, Ibx=0, useFit=True):
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
    # directly load from the uncertainty normed data
    I1I1 = vc.Varr[0, power1, Ibx] * 1.0
    Q1Q1 = vc.Varr[1, power1, Ibx] * 1.0
    I2I2 = vc.Varr[2, power1, Ibx] * 1.0
    Q2Q2 = vc.Varr[3, power1, Ibx] * 1.0
    I1I2 = vc.Varr[4, power1, Ibx] * 1.0
    I1Q2 = vc.Varr[5, power1, Ibx] * 1.0
    Q1I2 = vc.Varr[6, power1, Ibx] * 1.0
    Q1Q2 = vc.Varr[7, power1, Ibx] * 1.0
    I1Q1 = 0.0  # This data is ignored for now.
    I2Q2 = 0.0

    # To convert to photon numbers at the input of the Hemt
    g1 = G1 * h * f1 * vc.B  # Norm. Factor
    g2 = G2 * h * f2 * vc.B
    g12 = np.sqrt(g1 * g2)
    a1 = kB * Tn1 / (2.0 * h * f1)  # Amp noise photons
    a2 = kB * Tn2 / (2.0 * h * f2)
    # n1 = uPi1 / 2.0
    # n2 = uPi2 / 2.0

    # Create Covariance matrix (includes uncertainty from data selection)
    covM = np.array([[I1I1 / g1 - a1, I1Q1 / g1, I1I2 / g12, I1Q2 / g12],
                     [I1Q1 / g1, Q1Q1 / g1 - a1, Q1I2 / g12, Q1Q2 / g12],
                     [I1I2 / g12, Q1I2 / g12, I2I2 / g2 - a2, I2Q2 / g2],
                     [I1Q2 / g12, Q1Q2 / g12, I2Q2 / g2, Q2Q2 / g2 - a2]])

    # Add uncertainty in Photon numbers of identity elements
    # Uamp = np.array([[n1, 0, 0, 0],
    #                  [0, n1, 0, 0],
    #                  [0, 0, n2, 0],
    #                  [0, 0, 0, n2]])
    # covM = covM + Uamp

    return covM


def rot_phase(CovM):
    a = CovM[0, 2] * 1.0 - CovM[1, 3] * 1.0
    b = 1.0 * 1j * (CovM[0, 3] + CovM[1, 2])
    Psi = a + b
    phase = np.angle(Psi)
    if phase is not 0.0:
        Psi = Psi*np.exp(-1j*phase)
    b = 0.0
    a = 0.0
    a = np.real(Psi)
    b = np.imag(Psi)
    CovM[0, 2] = a/2.0
    CovM[1, 3] = a/2.0
    CovM[0, 3] = b/2.0
    CovM[1, 2] = b/2.0
    CovM[2, 0] = a/2.0
    CovM[3, 1] = a/2.0
    CovM[3, 0] = b/2.0
    CovM[2, 1] = b/2.0
    return CovM, phase


def NMatrix(vc, cpt=7, SnR=2):
    '''
    This assembles the Log neg matrix LnM
    '''
    vc.make_cvals(cpt, SnR)  # creates the necessary covariance values
    LnM = np.zeros([1, vc.d2.pt, vc.d3.pt])
    LnM2 = np.zeros([1, vc.d2.pt, vc.d3.pt])
    for ii in range(vc.d2.pt):
        for jj in range(vc.d3.pt):
            a = 1.0
            CovM = createCovMat(vc, ii, jj)
            CovM, phase = rot_phase(CovM)
            N = get_LogNegNum(CovM)
            Nsq = get_sqIneq(vc, CovM)
            if N == 0 or Nsq == 0.0:
                a = -1.0
            LnM[0, ii, jj] = N * a
            LnM2[0, ii, jj] = Nsq * a

    return LnM, LnM2


vc = sn.variable_carrier()
vc.fifolder = '1206//'
# vc.LP = 1.2
vc.LP = 2.9
vc.Texp = 0.012
SNR = 4.0

# Prev extracted data from 952_...
f1 = 4.1e9
f2 = 4.8e9
G1 = 2.6002e7  # Gain f1
G2 = 3.7766e+7  #
Tn1 = 10.79    # Noise Temp Amplifier (2K without filters)
Tn2 = 7.19
uG2 = 0.03e7    # Uncertainty values
uG2 = 0.002e7
uTn1 = 0.03
uTn2 = 0.02
uPi1 = 0.15  # resulting Photon uncertainty at Hemt input
uPi2 = 0.09

vc.LP = 1.0
folder1 = 'on12//'
vc.filein1 = folder1 + '1206CovMat_cI1I1.mtx'
vc.filein2 = folder1 + '1206CovMat_cQ1Q1.mtx'
vc.filein3 = folder1 + '1206CovMat_cI2I2.mtx'
vc.filein4 = folder1 + '1206CovMat_cQ2Q2.mtx'
vc.filein6 = folder1 + '1206CovMat_cI1I2.mtx'
vc.filein7 = folder1 + '1206CovMat_cI1Q2.mtx'
vc.filein8 = folder1 + '1206CovMat_cQ1I2.mtx'
vc.filein9 = folder1 + '1206CovMat_cQ1Q2.mtx'
vc.filein10 = folder1 + '1206CovMat_cI1Q1.mtx'
vc.filein11 = folder1 + '1206CovMat_cI2Q2.mtx'
vc.filein5 = '1206Vx1k.mtx'
vc.RTR = 1012 * 1e3  # RT resistor for
vc.load_and_go()
LnM, LnM2 = NMatrix(vc, cpt=4, SnR=2)
savename = '1206_LogN.mtx'
savename2 = '1206_ineqSq.mtx'
headtxt = make_header(vc.d3, vc.d2, vc.d1, meas_data='Log-Negativity')
savemtx(savename, LnM, headtxt)
headtxt2 = make_header(vc.d3, vc.d2, vc.d1, meas_data='Squeezing-Ineq')
savemtx(savename2, LnM2, headtxt2)
