import numpy as np
from scipy.constants import h, e
from scipy.constants import Boltzmann as kB
import matplotlib.pyplot as plt
import SNfit2 as SNfit
import pickle
# plt.ion()

vc = SNfit.variable_carrier()
vc.LP = 0.1
vc.T = 0.007
vc.Tn1 = 2.9
vc.Tn2 = 3.6
vc.G1 = 6.45e9
vc.G2 = 7.5e9
SNR = 4.0
f1 = 4.8e9
f2 = 4.1e9
vc.Ravg = 69.7
vc.B = 1e5
vc.inclMin = 2  # (how many smallest point values to include in the fit)
vc.resultfolder = '1150_SN1//'

savename = '1150_SN1_LogN.mtx'
savename2 = '1150_SN1_ineqSq.mtx'
vc.fifolder = '//Volumes//QDP-Backup-2//BenS//DCE2015-16//data_May20//'
subfolder = '1150SN1_ON//'
vc.filein1 = subfolder + '1150SN1_CovMat_cI1I1.mtx'
vc.filein2 = subfolder + '1150SN1_CovMat_cQ1Q1.mtx'
vc.filein3 = subfolder + '1150SN1_CovMat_cI2I2.mtx'
vc.filein4 = subfolder + '1150SN1_CovMat_cQ2Q2.mtx'
vc.filein6 = subfolder + '1150SN1_CovMat_cI1I2.mtx'
vc.filein7 = subfolder + '1150SN1_CovMat_cI1Q2.mtx'
vc.filein8 = subfolder + '1150SN1_CovMat_cQ1I2.mtx'
vc.filein9 = subfolder + '1150SN1_CovMat_cQ1Q2.mtx'
vc.filein10 = subfolder + '1150SN1_CovMat_cI1Q1.mtx'
vc.filein11 = subfolder + '1150SN1_CovMat_cI2Q2.mtx'
vc.filein5 = '1150SN1_Vx1k.mtx'
vc.RTR = 1012 * 1e3  # RT resistor for
vc.load_and_go()

# create crop vector for the fitting
crop_within = SNfit.find_nearest(vc.I, -6.0e-6), SNfit.find_nearest(vc.I, 6.0e-6)
print 'crop_within', crop_within
crop_outside = SNfit.find_nearest(vc.I, -19.0e-6), SNfit.find_nearest(vc.I, 20.0e-6)
print 'crop_outside', crop_outside
vc.crop = [crop_within, crop_outside]
print 'vc.crop', vc.crop

snd, r1, r2, r3, r4 = SNfit.DoSNfits(vc, True)  # run fits

# plt.show()
