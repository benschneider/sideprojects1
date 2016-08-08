import numpy as np
from scipy.constants import h, e
from scipy.constants import Boltzmann as kB
import SNfit2 as SNfit  # SNfit (traditionally) & SNfit2 (for hybrid coupler)
# import PyGnuplot as gp

vc = SNfit.variable_carrier()
vc.fifolder = ''
vc.LP = 1.0
vc.Texp = 0.009
SNR = 4.0
f1 = 4.8e9
f2 = 4.8e9
vc.Ravg = 0.0  # 69.7
vc.B = 1e5
vc.resultfolder = 'f2//'

pathf1 = '/Volumes/QDP-Backup-2/BenS/DCE2015-16/data_Jul15/1206_SN22_1/'
savename = '1150_LogN.mtx'
savename2 = '1150_ineqSq.mtx'
vc.fifolder = ''
vc.filein1 = pathf1 + '1206_SN22_1ON//1206_SN22_1CovMat_cI1I1.mtx'
vc.filein2 = pathf1 + '1206_SN22_1ON//1206_SN22_1CovMat_cQ1Q1.mtx'
vc.filein3 = pathf1 + '1206_SN22_1ON//1206_SN22_1CovMat_cI2I2.mtx'
vc.filein4 = pathf1 + '1206_SN22_1ON//1206_SN22_1CovMat_cQ2Q2.mtx'
vc.filein6 = pathf1 + '1206_SN22_1ON//1206_SN22_1CovMat_cI1I2.mtx'
vc.filein7 = pathf1 + '1206_SN22_1ON//1206_SN22_1CovMat_cI1Q2.mtx'
vc.filein8 = pathf1 + '1206_SN22_1ON//1206_SN22_1CovMat_cQ1I2.mtx'
vc.filein9 = pathf1 + '1206_SN22_1ON//1206_SN22_1CovMat_cQ1Q2.mtx'
vc.filein10 = pathf1 + '1206_SN22_1ON//1206_SN22_1CovMat_cI1Q1.mtx'
vc.filein11 = pathf1 + '1206_SN22_1ON//1206_SN22_1CovMat_cI2Q2.mtx'
vc.filein5 = pathf1 + '1206_SN22_1Vx1k.mtx'
vc.RTR = 1012 * 1e3  # RT resistor for
vc.load_and_go()

# create crop vector for the fitting
crop_within = SNfit.find_nearest(vc.I, -6.0e-6), SNfit.find_nearest(vc.I, 6.0e-6)
print 'crop_within', crop_within
crop_outside = SNfit.find_nearest(vc.I, -19e-6), SNfit.find_nearest(vc.I, 20e-6)
print 'crop_outside', crop_outside
vc.crop = [crop_within, crop_outside]
print 'vc.crop', vc.crop

snd, r1, r2, r3, r4 = SNfit.DoSNfits(vc, True)  # run fits

