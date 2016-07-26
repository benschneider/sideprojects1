import numpy as np
from scipy.constants import h, e
from scipy.constants import Boltzmann as kB
import matplotlib.pyplot as plt
import SNfit
plt.ion()

vc = SNfit.variable_carrier()
vc.fifolder = ''
vc.LP = 0.1
vc.Texp = 0.010
SNR = 4.0
f1 = 4.8e9
f2 = 4.1e9
vc.Ravg = 71.9
vc.B = 1e5

savename = '1150_SN1_LogN.mtx'
savename2 = '1150_SN1_ineqSq.mtx'
vc.fifolder = ''
vc.filein1 = '1150SN1_ON//1150SN1_CovMat_cI1I1.mtx'
vc.filein2 = '1150SN1_ON//1150SN1_CovMat_cQ1Q1.mtx'
vc.filein3 = '1150SN1_ON//1150SN1_CovMat_cI2I2.mtx'
vc.filein4 = '1150SN1_ON//1150SN1_CovMat_cQ2Q2.mtx'
vc.filein5 = '1150SN1_Vx1k.mtx'
vc.filein6 = '1150SN1_ON//1150SN1_CovMat_cI1I2.mtx'
vc.filein7 = '1150SN1_ON//1150SN1_CovMat_cI1Q2.mtx'
vc.filein8 = '1150SN1_ON//1150SN1_CovMat_cQ1I2.mtx'
vc.filein9 = '1150SN1_ON//1150SN1_CovMat_cQ1Q2.mtx'
vc.filein10 = '1150SN1_ON//1150SN1_CovMat_cI1Q1.mtx'
vc.filein11 = '1150SN1_ON//1150SN1_CovMat_cI2Q2.mtx'
vc.RTR = 1012 * 1e3  # RT resistor for
vc.load_and_go()

# create crop vector for the fitting
crop_within = SNfit.find_nearest(vc.I, -8.01e-6), SNfit.find_nearest(vc.I, 8.01e-6)
print 'crop_within', crop_within
crop_outside = SNfit.find_nearest(vc.I, -20e-6), SNfit.find_nearest(vc.I, 20e-6)
print 'crop_outside', crop_outside
vc.crop = [crop_within, crop_outside]
print 'vc.crop', vc.crop

snd = SNfit.DoSNfits(vc, True)  # run fits
# plt.show()
