from parsers import make_header, savemtx
import numpy as np
import SNfit2 as sn
from scipy.constants import h, e
from scipy.constants import Boltzmann as kB


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
    if C == 0:
        return 0.0

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
    n1 = (CovM[0, 0] * 1.0 + CovM[1, 1] * 1.0 - 0.5)  # excluding zpf
    n2 = (CovM[2, 2] * 1.0 + CovM[3, 3] * 1.0 - 0.5)  # excluding zpf
    # Photons detected to be TMS:
    sqp1 = CovM[0, 2] * 1.0 - CovM[1, 3] * 1.0 + 1.0 * 1j * (CovM[0, 3] + CovM[1, 2])
    squeezing = np.abs(sqp1) / ((n1 + n2 + 1) / 2.0)  # squeezing includes zpf again
    a = 2.0*np.sqrt(vc.f1*vc.f2)*(n1+n2)
    b = vc.f1*(2.0*n1+1.0) + vc.f2*(2.0*n2+1.0)
    ineq = a/b
    return squeezing, ineq, n1, n2


def TwoModeSqueeze_inequality(f1, f2, n):
    ''' This function calculates the amount of two mode
    squeezing required to prove hard entanglement
    '''
    return ((2.0 * np.sqrt(f1 * f2) * (2 * n)) / (f1 * (2 * n + 1) + f2 * (2 * n + 1)))


def rot_phase(CovMin):
    CovM = CovMin * 1.0
    a = CovM[0, 2] * 1.0 - CovM[1, 3] * 1.0
    b = 1.0 * (CovM[0, 3] + CovM[1, 2])
    Psi = a + 1j*b
    phase = np.angle(Psi)
    if phase is not 0.0:
        a = np.abs(Psi)  # set angle to zero
        b = 0

    CovM[0, 2] = a/2.0
    CovM[1, 3] = a/2.0
    CovM[0, 3] = b/2.0
    CovM[1, 2] = b/2.0
    CovM[2, 0] = a/2.0
    CovM[3, 1] = a/2.0
    CovM[3, 0] = b/2.0
    CovM[2, 1] = b/2.0
    return CovM, phase


def createCovMat(vc, vc2, power1=0, Ibx=0):
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
    I1I1_2 = vc2.Varr[0, power1, Ibx] * 1.0  # Background values when drive is switched off
    Q1Q1_2 = vc2.Varr[1, power1, Ibx] * 1.0
    I2I2_2 = vc2.Varr[2, power1, Ibx] * 1.0
    Q2Q2_2 = vc2.Varr[3, power1, Ibx] * 1.0
    # To convert to photon numbers at the input of the Hemt
    g1 = G1 * h * vc.f1 * vc.B  # Norm. Factor
    g2 = G2 * h * vc.f2 * vc.B
    g12 = np.sqrt(g1 * g2)

    ncc = (abs(vc.Narr[4, power1, Ibx] * 1.0) +  # uncertainty of all cross correlation values
           abs(vc.Narr[5, power1, Ibx] * 1.0) +
           abs(vc.Narr[6, power1, Ibx] * 1.0) +
           abs(vc.Narr[7, power1, Ibx] * 1.0))

    n00 = abs(vc.Narr[0, power1, Ibx] * 1.0) # Noise Variance
    n11 = abs(vc.Narr[1, power1, Ibx] * 1.0)
    n22 = abs(vc.Narr[2, power1, Ibx] * 1.0)
    n33 = abs(vc.Narr[3, power1, Ibx] * 1.0)

    # Create Covariance matrix (includes uncertainty from data selection)
    covM = np.array([[(I1I1-I1I1_2) / g1 + 0.25, I1Q1 / g1, I1I2 / g12, I1Q2 / g12],
                     [I1Q1 / g1, (Q1Q1-Q1Q1_2) / g1 + 0.25, Q1I2 / g12, Q1Q2 / g12],
                     [I1I2 / g12, Q1I2 / g12, (I2I2-I2I2_2) / g2 + 0.25, I2Q2 / g2],
                     [I1Q2 / g12, Q1Q2 / g12, I2Q2 / g2, (Q2Q2-Q2Q2_2) / g2 + 0.25]])

    # Add uncertainty in Photon numbers of identity elements
    Uamp = np.array([[n00, 0, 0, 0],
                     [0, n11, 0, 0],
                     [0, 0, n22, 0],
                     [0, 0, 0, n33]])
    # covM = covM + Uamp
    return covM


def NMatrix(vc, vc2):
    '''
    This assembles the Log neg matrix LnM
    '''
    LnM = np.zeros([vc.d2.pt, vc.d3.pt])
    LnM2 = np.zeros([vc.d2.pt, vc.d3.pt])
    n1 = np.zeros([vc.d2.pt, vc.d3.pt])
    n2 = np.zeros([vc.d2.pt, vc.d3.pt])
    Ineq = np.zeros([vc.d2.pt, vc.d3.pt])
    for ii in range(vc.d2.pt):
        for jj in range(vc.d3.pt):
            a = 1.0
            CovM = createCovMat(vc, vc2, ii, jj)
            CovM, phase = rot_phase(CovM)
            N = get_LogNegNum(CovM)
            Nsq, Ineq[ii, jj], n1[ii, jj], n2[ii, jj] = get_sqIneq(vc, CovM)
            a = 1.0
            LnM[ii, jj] = N * a
            LnM2[ii, jj] = Nsq * a
    return LnM, LnM2, Ineq, n1, n2

fname = '1150_'
vc = sn.variable_carrier()
vc.resultfolder = fname+'//'
folder = '/Volumes/QDP-Backup-2/BenS/DCE2015-16/data_May20/'
vc.fifolder = folder
vc.snr = 4.5
vc.cpt = 5
vc.LP = 0.0
vc.T = 0.007
vc.Tn1 = 3.9
vc.Tn2 = 4.6
vc.G1 = 6.45e9
vc.G2 = 7.5e9
vc.f1 = 4.8e9
vc.f2 = 4.1e9
vc.Ravg = 0.0  # 69.7  # if set to 0.0 script will use differential resistance
vc.B = 1e5
vc.inclMin = 10  # if set to 0 will not add values close to 0 Ib
folder1 = fname + 'ON//'
vc.filein1 = folder1 + fname + 'CovMat_cI1I1.mtx'
vc.filein2 = folder1 + fname + 'CovMat_cQ1Q1.mtx'
vc.filein3 = folder1 + fname + 'CovMat_cI2I2.mtx'
vc.filein4 = folder1 + fname + 'CovMat_cQ2Q2.mtx'
vc.filein6 = folder1 + fname + 'CovMat_cI1I2.mtx'
vc.filein7 = folder1 + fname + 'CovMat_cI1Q2.mtx'
vc.filein8 = folder1 + fname + 'CovMat_cQ1I2.mtx'
vc.filein9 = folder1 + fname + 'CovMat_cQ1Q2.mtx'
vc.filein10 = folder1 + fname + 'CovMat_cI1Q1.mtx'
vc.filein11 = folder1 + fname + 'CovMat_cI2Q2.mtx'
vc.filein5 = fname + 'Vx1k.mtx'
vc.RTR = 1012 * 1e3  # RT resistor for
vc.load_and_go()
crop_within = sn.find_nearest(vc.I, -6.0e-6), sn.find_nearest(vc.I, 6.0e-6)
crop_outside = sn.find_nearest(vc.I, -19e-6), sn.find_nearest(vc.I, 20e-6)
vc.crop = [crop_within, crop_outside]

fname2 = '1150_'
vc2 = sn.variable_carrier()
vc2.resultfolder = fname2+'//'
folder = '/Volumes/QDP-Backup-2/BenS/DCE2015-16/data_May20/'
vc2.fifolder = folder
vc2.snr = 4.5
vc2.cpt = 5
vc2.LP = 0.0
vc2.T = 0.007
vc2.Tn1 = 3.9
vc2.Tn2 = 4.6
vc2.G1 = 6.45e9
vc2.G2 = 7.5e9
vc2.f1 = 4.8e9
vc2.f2 = 4.1e9
vc2.Ravg = 0.0  # 69.7  # if set to 0.0 script will use differential resistance
vc2.B = 1e5
vc2.inclMin = 10  # if set to 0 will not add values close to 0 Ib
folder1 = fname2 + 'OFF/'
vc2.filein1 = folder1 + fname + 'CovMat_cI1I1.mtx'
vc2.filein2 = folder1 + fname + 'CovMat_cQ1Q1.mtx'
vc2.filein3 = folder1 + fname + 'CovMat_cI2I2.mtx'
vc2.filein4 = folder1 + fname + 'CovMat_cQ2Q2.mtx'
vc2.filein6 = folder1 + fname + 'CovMat_cI1I2.mtx'
vc2.filein7 = folder1 + fname + 'CovMat_cI1Q2.mtx'
vc2.filein8 = folder1 + fname + 'CovMat_cQ1I2.mtx'
vc2.filein9 = folder1 + fname + 'CovMat_cQ1Q2.mtx'
vc2.filein10 = folder1 + fname + 'CovMat_cI1Q1.mtx'
vc2.filein11 = folder1 + fname + 'CovMat_cI2Q2.mtx'
vc2.filein5 = fname + 'Vx1k.mtx'
vc2.RTR = 1012 * 1e3  # RT resistor for
vc2.load_and_go()

G1 = 601.0e7
G2 = 685.35e7
Tn1 = 3.51
Tn2 = 4.38

num0 = 17
BLnM = np.zeros([num0, vc.d2.pt, vc.d3.pt])
BLnM2 = np.zeros([num0, vc.d2.pt, vc.d3.pt])
Mn1 = np.zeros([num0, vc.d2.pt, vc.d3.pt])
Mn2 = np.zeros([num0, vc.d2.pt, vc.d3.pt])
MIneq = np.zeros([num0, vc.d2.pt, vc.d3.pt])

# LnM, LnM2, Ineq, n1, n2
for i, vc.snr in enumerate(np.linspace(0, 8, num0)):
    vc.make_cvals()
    BLnM[i, :, :], BLnM2[i, :, :], MIneq[i, :, :], Mn1[i, :, :], Mn2[i, :, :] = NMatrix(vc, vc2)
vc.d1.name = 'Signal/Noise'
vc.d1.pt = num0
vc.d1.start = 0.0
vc.d1.stop = 8.0

# convert to effective Phi0
vc.d3.start = vc.d3.start*0.0113
vc.d3.stop = vc.d3.stop*0.0113

ext = 'med_'

savename3 = fname+ext+'n1.mtx'
headtxt3 = make_header(vc.d3, vc.d2, vc.d1, meas_data='n1')
savemtx(savename3, Mn1, headtxt3)

savename4 = fname+ext+'n2.mtx'
headtxt4 = make_header(vc.d3, vc.d2, vc.d1, meas_data='n2')
savemtx(savename4, Mn2, headtxt4)

savename5 = fname+ext+'IneQ.mtx'
headtxt5 = make_header(vc.d3, vc.d2, vc.d1, meas_data='Ineq')
savemtx(savename5, MIneq, headtxt5)

savename = fname+ext+'LogN.mtx'
headtxt = make_header(vc.d3, vc.d2, vc.d1, meas_data='Log-Negativity')
savemtx(savename, BLnM, headtxt)

savename2 = fname+ext+'ineqSq.mtx'
headtxt2 = make_header(vc.d3, vc.d2, vc.d1, meas_data='Squeezing-Ineq')
savemtx(savename2, BLnM2, headtxt2)
