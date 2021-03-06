import numpy as np
from parsers import storehdf5  # , loadmtx, savemtx
from scipy.constants import h  # , e, c , pi
from scipy.signal.signaltools import _next_regular
from scipy.ndimage.filters import gaussian_filter1d
from numpy.fft import rfftn, irfftn
import logging
from multiprocessing import Process, Queue, Pipe
from time import time
import PyGnuplot as gp
import gc


def load_dataset(dispData, dicData, ext='hdf5_on'):
	'''Reads from the dictionary which data set was selected'''
	dicData[ext] = storehdf5(str(dispData[ext]))
	dicData[ext].open_f(mode='r')


def f1pN2(tArray, lags0, d=1):
	squeezing_noise = np.sqrt(np.var(np.abs(tArray)))  # including the peak matters little
	if np.max(np.abs(tArray[lags0 - d:lags0 + d + 1])) < 3.0 * squeezing_noise:
		logging.debug('SN ratio too low: Can not find trigger position')
		distance = 0
	else:
		distance = (np.argmax(tArray[lags0 - d:lags0 + d + 1]) - d) * -1
	return distance


def f1pN(tArray, lags0, d=1):
	return (np.argmax(tArray[lags0 - d:lags0 + d + 1]) - d) * -1


def process(dispData, dicData):
	assignRaw(dispData, dicData)
	makeheader(dispData, dicData)
	get_data_avg2(dispData, dicData)


def process_all_points(dispData, dicData):
	assignRaw(dispData, dicData)
	makeheader(dispData, dicData)
	lags = dispData['lags']
	mapdim = dispData['mapdim']
	start_select = dispData['select']
	pt = int(dispData['Process Num'] / dispData['Power Averages'])
	res = dicData['res']
	res.IQmapMs_avg = np.zeros([pt, 4, mapdim[0], mapdim[1]])
	res.cs_avg = np.zeros([pt, 10, lags * 2 + 1])
	res.cs_avg_off = np.zeros([pt, 10, lags * 2 + 1])
	res.ns = np.zeros([pt, 2])
	res.ns_off = np.zeros([pt, 2])
	res.ineqs = np.zeros(pt)
	res.sqs = np.zeros(pt)
	res.sqs2 = np.zeros(pt)
	res.sqsn2 = np.zeros(pt)
	res.sqphs = np.zeros(pt)
	res.noises = np.zeros(pt)
	for n in range(pt):
		dispData['select'] = start_select + n * dispData['Power Averages']
		get_data_avg2(dispData, dicData)
		res.IQmapMs_avg[n] = res.IQmapM_avg
		res.cs_avg[n] = res.c_avg
		res.cs_avg_off[n] = res.c_avg_off
		res.ns[n] = res.n
		res.ns_off[n][0] = res.covMatOff[0, 0] + res.covMatOff[1, 1]
		res.ns_off[n][1] = res.covMatOff[2, 2] + res.covMatOff[3, 3]
		res.sqs[n] = res.sq
		res.sqs2[n] = res.psi_mag_avg[0]
		res.sqsn2[n] = res.psi_mag_avg[1]
		res.sqphs[n] = res.sqph
		res.ineqs[n] = res.ineq
		res.noises[n] = res.noise
		logging.debug('SQ:' + str(res.sq) + 'INEQ:' + str(res.ineq))


def assignRaw(dispData, dicData):
	on, off, Fac1, Fac2 = prep_data(dispData, dicData)
	select = dispData['select']
	d1 = on.h5.root
	LP = dispData['Low Pass']
	d2 = off.h5.root
	on.I1 = lowPass(d1.D12raw[select][0] / Fac1, lowpass_sigma=LP)
	on.Q1 = lowPass(d1.D12raw[select][1] / Fac1, lowpass_sigma=LP)
	on.I2 = lowPass(d1.D12raw[select][2] / Fac2, lowpass_sigma=LP)
	on.Q2 = lowPass(d1.D12raw[select][3] / Fac2, lowpass_sigma=LP)
	off.I1 = lowPass(d2.D12raw[select][0] / Fac1, lowpass_sigma=LP)
	off.Q1 = lowPass(d2.D12raw[select][1] / Fac1, lowpass_sigma=LP)
	off.I2 = lowPass(d2.D12raw[select][2] / Fac2, lowpass_sigma=LP)
	off.Q2 = lowPass(d2.D12raw[select][3] / Fac2, lowpass_sigma=LP)
	dispData['I,Q data length'] = int(len(on.I1))
	dispData['Trace i, j, k'] = list(d1.ijk[select])


def prep_data(dispData, dicData):
	on = dicData['hdf5_on']
	off = dicData['hdf5_off']
	f1 = dispData['f1']
	f2 = dispData['f2']
	G1 = dispData['g1']
	G2 = dispData['g2']
	B = dispData['B']
	Fac1 = np.sqrt(h * f1 * G1 * B)
	Fac2 = np.sqrt(h * f2 * G2 * B)
	return on, off, Fac1, Fac2


def lowPass(data, lowpass_sigma=0.0):
	if bool(lowpass_sigma):
		data = gaussian_filter1d(data, lowpass_sigma)  # Gausfilter
	return data - np.mean(data)  # remove DC offset


def mp_wrap(pin, function, *args2):
	''' retvalues are the returns  and function is the function that gives you these'''
	retvalues = function(*args2)
	pin.send(retvalues)
	pin.close()


def makehist2d(dispData, dicData):
	gc.collect()
	on = dicData['hdf5_on']
	off = dicData['hdf5_off']
	t0 = time()
	res = dicData['res']
	on.IImap, on.xII, on.yII = np.histogram2d(on.I1, on.I2, [on.xII, on.yII])
	on.QQmap, on.xQQ, on.yQQ = np.histogram2d(on.Q1, on.Q2, [on.xQQ, on.yQQ])
	on.IQmap, on.xIQ, on.yIQ = np.histogram2d(on.I1, on.Q2, [on.xIQ, on.yIQ])
	on.QImap, on.xQI, on.yQI = np.histogram2d(on.Q1, on.I2, [on.xQI, on.yQI])
	off.IImap, off.xII, off.yII = np.histogram2d(off.I1, off.I2, [on.xII, on.yII])
	off.QQmap, off.xQQ, off.yQQ = np.histogram2d(off.Q1, off.Q2, [on.xQQ, on.yQQ])
	off.IQmap, off.xIQ, off.yIQ = np.histogram2d(off.I1, off.Q2, [on.xIQ, on.yIQ])
	off.QImap, off.xQI, off.yQI = np.histogram2d(off.Q1, off.I2, [on.xQI, on.yQI])
	res.IQmapM[0] = on.IImap - off.IImap
	res.IQmapM[1] = on.QQmap - off.QQmap
	res.IQmapM[2] = on.IQmap - off.IQmap
	res.IQmapM[3] = on.QImap - off.QImap
	t1 = time()
	logging.debug('End MP histogramm time used: ' + str(t1 - t0))


def makeheader(dispData, dicData):
	assignRaw(dispData, dicData)
	on = dicData['hdf5_on']
	res = dicData['res']
	mapdim = dispData['mapdim']
	res.IQmapM = np.zeros([4, mapdim[0], mapdim[1]])
	res.IQmapM[2], on.xIQ, on.yIQ = np.histogram2d(on.I1, on.Q2, mapdim)
	res.IQmapM[2], on.xIQ, on.yIQ = np.histogram2d(on.I1, on.Q2, [on.xIQ, on.xIQ])  # want them all equally binned
	res.IQmapM[0], on.xII, on.yII = np.histogram2d(on.I1, on.I2, [on.xIQ, on.yIQ])
	res.IQmapM[1], on.xQQ, on.yQQ = np.histogram2d(on.Q1, on.Q2, [on.xIQ, on.yIQ])
	res.IQmapM[3], on.xQI, on.yQI = np.histogram2d(on.Q1, on.I2, [on.xIQ, on.yIQ])
	endstr = ',' + str(dispData['dim1 name']) + ',' + str(dispData['dim1 start']) + ',' + str(dispData['dim1 stop'])
	on.headerII = ('Units,bin,I1,' + str(on.xII[0]) + ',' + str(on.xII[-2]) +
	               ',I2,' + str(on.yII[0]) + ',' + str(on.yII[-2]) + endstr)
	on.headerQQ = ('Units,bin,Q1,' + str(on.xQQ[0]) + ',' + str(on.xQQ[-2]) +
	               ',Q2,' + str(on.yQQ[0]) + ',' + str(on.yQQ[-2]) + endstr)
	on.headerIQ = ('Units,bin,I1,' + str(on.xIQ[0]) + ',' + str(on.xIQ[-2]) +
	               ',Q2,' + str(on.yIQ[0]) + ',' + str(on.yIQ[-2]) + endstr)
	on.headerQI = ('Units,bin,Q1,' + str(on.xQI[0]) + ',' + str(on.xQI[-2]) +
	               ',I2,' + str(on.yQI[0]) + ',' + str(on.yQI[-2]) + endstr)


def getCovMatrix(IQdata, lags=100, hp=False):
	# 0: <I1I2> # 1: <Q1Q2> # 2: <I1Q2> # 3: <Q1I2> # 4: <Squeezing> Magnitude # 5: <Squeezing> Phase
	lags = int(lags)
	I1 = np.asarray(IQdata[0])
	Q1 = np.asarray(IQdata[1])
	I2 = np.asarray(IQdata[2])
	Q2 = np.asarray(IQdata[3])
	CovMat = np.zeros([10, lags * 2 + 1])
	start = len(I1) - lags - 1
	stop = len(I1) + lags
	sI1 = np.array(I1.shape)
	shape0 = sI1 * 2 - 1
	fshape = [_next_regular(int(d)) for d in shape0]  # padding to optimal size for FFTPACK
	fslice = tuple([slice(0, int(sz)) for sz in shape0])
	# Do FFTs and get Cov Matrix
	fftI1 = rfftn(I1, fshape)
	fftQ1 = rfftn(Q1, fshape)
	fftI2 = rfftn(I2, fshape)
	fftQ2 = rfftn(Q2, fshape)
	rfftI1 = rfftn(I1[::-1], fshape)  # there should be a simple relationship to fftI1
	rfftQ1 = rfftn(Q1[::-1], fshape)
	rfftI2 = rfftn(I2[::-1], fshape)
	rfftQ2 = rfftn(Q2[::-1], fshape)
	CovMat[0, :] = irfftn((fftI1 * rfftI2))[fslice].copy()[start:stop] / fshape
	CovMat[1, :] = irfftn((fftQ1 * rfftQ2))[fslice].copy()[start:stop] / fshape
	CovMat[2, :] = irfftn((fftI1 * rfftQ2))[fslice].copy()[start:stop] / fshape
	CovMat[3, :] = irfftn((fftQ1 * rfftI2))[fslice].copy()[start:stop] / fshape
	psi = (1j * (CovMat[2, :] + CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))
	CovMat[4, :] = abs(psi)
	CovMat[5, :] = np.angle(psi)
	CovMat[6, :] = irfftn((fftI1 * rfftI1))[fslice].copy()[start:stop] / fshape
	CovMat[7, :] = irfftn((fftQ1 * rfftQ1))[fslice].copy()[start:stop] / fshape
	CovMat[8, :] = irfftn((fftI2 * rfftI2))[fslice].copy()[start:stop] / fshape
	CovMat[9, :] = irfftn((fftQ2 * rfftQ2))[fslice].copy()[start:stop] / fshape
	return CovMat


# Segmenting the data does not help ! -> no wrapper needed
# def getCovMat_wrap(dispData, data):
#     # segment = dispData['Segment Size']
#     lags = dispData['lags']
#     hp = dispData['FFT-Filter']
#     IQdata = np.array([data.I1, data.Q1, data.I2, data.Q2])
#     # if segment:
#     #     num = len(IQdata[0]) / segment
#     #     modulo = len(IQdata[0]) % segment  # 4/3 = 1 + residual -> would give an error
#     #     if bool(modulo):
#     #         for n in range(IQdata.shape[0]):
#     #             IQdata[n] = IQdata[n][:-modulo]
#     #     # for n in range(IQdata.shape[0]):
#     #     IQdata2 = np.reshape(IQdata[:], [IQdata.shape[0], num, segment])
#     #     CovMat = np.zeros([10, dispData['lags'] * 2 + 1])
#     #     for i in range(num):
#     #         CovMat += getCovMatrix(IQdata2[:, i], lags=lags, hp=hp)
#     #     CovMat = CovMat / np.float(num)
#     #     psi = (1j * (CovMat[2, :] + CovMat[3, :]) + (CovMat[0, :] - CovMat[1, :]))
#     #     CovMat[4, :] = np.abs(psi)
#     #     CovMat[5, :] = np.angle(psi)
#     # else:
#     CovMat = getCovMatrix(IQdata, lags=lags, hp=hp)
#     return CovMat


def correctPhase(dispData, dicData, avg_phase_offset=0.0):
	on, off, Fac1, Fac2 = prep_data(dispData, dicData)
	IQdata_on = np.array([on.I1, on.Q1, on.I2, on.Q2])
	IQdata_off = np.array([off.I1, off.Q1, off.I2, off.Q2])
	dMag = 0
	if dispData['Trigger correction']:
		CovMat = getCovMatrix(IQdata_on, lags=dispData['lags'])
		dMag = f1pN(CovMat[4], dispData['lags'], d=1)
		on.I1 = np.roll(on.I1, dMag)  # Correct 1pt trigger jitter
		on.Q1 = np.roll(on.Q1, dMag)
		logging.debug('Trigger corrected ' + str(dMag) + 'pt')
	if dispData['Phase correction']:
		phase_index = dispData['lags'] + dMag # np.argmax(CovMat[4])
		phase_offset = CovMat[5][phase_index] + avg_phase_offset
		phase = np.angle(1j * on.Q1 + on.I1)  # phase rotation
		new = np.abs(1j * on.Q1 + on.I1) * np.exp(1j * (phase - phase_offset))
		on.I1 = np.real(new)
		on.Q1 = np.imag(new)
		logging.debug('Phase corrected ' + str(phase_offset) + 'rad')
	IQdata_on = np.array([on.I1, on.Q1, on.I2, on.Q2])
	IQdata_off = np.array([off.I1, off.Q1, off.I2, off.Q2])
	on.CovMat = getCovMatrix(IQdata_on, lags=dispData['lags'])
	# off.CovMat = getCovMat_wrap(dispData, off)


def get_phase_trigger_cov(on, phase_offset2=0.0, lags=1000):
	IQdata_on = np.array([on.I1, on.Q1, on.I2, on.Q2])
	CovMat = getCovMatrix(IQdata_on, lags=lags)
	dMag = f1pN2(CovMat[4], lags, d=1)
	on.I1 = np.roll(on.I1, dMag)  # Correct 1pt trigger jitter
	on.Q1 = np.roll(on.Q1, dMag)
	phase_index = lags + dMag
	phase_offset = CovMat[5][phase_index] + phase_offset2
	phase = np.angle(1j * on.Q1 + on.I1)  # phase rotation
	new = np.abs(1j * on.Q1 + on.I1) * np.exp(1j * (phase - phase_offset))
	on.I1 = np.real(new)
	on.Q1 = np.imag(new)
	return phase_offset, dMag


def correct_phase_trigger(on, phase_offset, trigger):
	on.I1 = np.roll(on.I1, int(trigger))  # Correct 1pt trigger jitter
	on.Q1 = np.roll(on.Q1, int(trigger))
	phase = np.angle(1j * on.Q1 + on.I1)
	new = np.abs(1j * on.Q1 + on.I1) * np.exp(1j * (phase - phase_offset))
	on.I1 = np.real(new)
	on.Q1 = np.imag(new)
	logging.debug('Phase, Trigger corrected:' + str(phase_offset) + ' rad , ' +str(trigger) + ' pt')


def get_data_avg2(dispData, dicData):
	res = dicData['res']
	lags = dispData['lags']
	res.c_avg = np.zeros([10, lags * 2 + 1])                    # Covariance Map inc PSI
	res.c_avg_off = np.zeros([10, lags * 2 + 1])                # Covariance Map
	res.psi_avg = 1j * np.zeros([1, lags * 2 + 1])              # PSI
	res.psi_mag_avg = np.zeros(2)
	on = dicData['hdf5_on']
	off = dicData['hdf5_off']
	phase_list = np.zeros([dispData['dim1 pt']*dispData['dim2 pt']])
	trigger_list = np.zeros_like(phase_list)
	xavg = (dispData['Averages'])
	yavg = (dispData['Power Averages'])
	selected = dispData['select']
	covMat2 = np.zeros([4, 4])
	phase_offset1 = 0.0
	phase_offset2 = 0.0
	for j in range(yavg):
		covMat1 = np.zeros([4, 4])
		for i in range(xavg):
			idx = selected + j + i*dispData['dim1 pt']
			dispData['select'] = idx
			assignRaw(dispData, dicData)  # load next 'on' I,Q data
			phase_list[idx], trigger_list[idx] = get_phase_trigger_cov(on, phase_offset2=0.0, lags=lags)
			covMat1 += np.cov([on.I1, on.Q1, on.I2, on.Q2])  # average fist covariance matrixes
		covMat1 /= xavg
		phase_offset1 = np.angle( (covMat1[2, 0] - covMat1[3, 1]) + 1j * (covMat1[3, 0] + covMat1[2, 1]))  # averaged phase offset
		# apply phase_offset1 to phase list
		covMat1 = np.zeros([4, 4])
		for i in range(xavg):
			idx = selected + j + i*dispData['dim1 pt']
			phase_list[idx] += phase_offset1
			dispData['select'] = idx
			assignRaw(dispData, dicData)  # load next 'on' I,Q data
			correct_phase_trigger(on, phase_list[idx], trigger_list[idx])
			covMat1 += np.cov([on.I1, on.Q1, on.I2, on.Q2])  # average phase averaged corrected matrix
		covMat1 /= xavg
		phase_offset1 = np.angle( (covMat1[2, 0] - covMat1[3, 1]) + 1j * (covMat1[3, 0] + covMat1[2, 1]))  # averaged phase offset
		logging.debug('residual phase shift 1:' + str(phase_offset1))
		covMat2 += covMat1  # average the phase averaged and corrected data 1
	covMat2 /= yavg
	phase_offset2 = np.angle( (covMat2[2, 0] - covMat2[3, 1]) + 1j * (covMat2[3, 0] + covMat2[2, 1]))  # averaged phase offset
	logging.debug('residual phase shift 2:' + str(phase_offset1))
	# now use both phase shift corrections
	covMat2 = np.zeros([4, 4])
	covMat2_off = np.zeros([4, 4])
	res.IQmapM_avg = np.zeros([4, dispData['mapdim'][0], dispData['mapdim'][1]])        # histogram map
	for j in range(yavg):
		covMat1 = np.zeros([4, 4])
		covMat1_off = np.zeros([4, 4])
		# covMat1_avg = np.zeros([4, 4])
		for i in range(xavg):
			idx = selected + j + i*dispData['dim1 pt']
			phase_list[idx] += phase_offset2
			dispData['select'] = idx
			assignRaw(dispData, dicData)                                    # load next 'on' I,Q data
			correct_phase_trigger(on, phase_list[idx], trigger_list[idx])
			correct_phase_trigger(off, phase_list[idx], trigger_list[idx])
			covMat1 += np.cov([on.I1, on.Q1, on.I2, on.Q2])                 # average corrected matrix (Drive ON)
			covMat1_off += np.cov([off.I1, off.Q1, off.I2, off.Q2])         # average corrected matrix (Drive OFF)
			# covMat1 -= covMat1_off                                           # subtract ON - OFF (req phase corrections)
			# covMat1_avg += covMat1  # average end result
			makehist2d(dispData, dicData)
			res.IQmapM_avg += res.IQmapM
			res.c_avg += getCovMatrix([on.I1, on.Q1, on.I2, on.Q2], lags=lags)
			res.psi_mag_avg += np.abs( (covMat1[2, 0] - covMat1[3, 1]) + 1j * (covMat1[3, 0] + covMat1[2, 1]))

		covMat1[0, 0] -= covMat1_off[0, 0]                              # subtract ON - OFF in photon numbers
		covMat1[0, 1] -= covMat1_off[0, 1]                              # subtract ON - OFF in photon numbers
		covMat1[1, 0] -= covMat1_off[1, 0]
		covMat1[1, 1] -= covMat1_off[1, 1]
		covMat1[2, 2] -= covMat1_off[2, 2]
		covMat1[2, 3] -= covMat1_off[2, 3]
		covMat1[3, 2] -= covMat1_off[3, 2]
		covMat1[3, 3] -= covMat1_off[3, 3]
		covMat1 /= xavg
		covMat1_off /= xavg
		phase_offset1 = np.angle( (covMat1[2, 0] - covMat1[3, 1]) + 1j * (covMat1[3, 0] + covMat1[2, 1]))
		logging.debug('residual phase shift 1 part 2:' + str(phase_offset1))
		covMat2 += covMat1  # average the phase averaged and corrected data 1
		covMat2_off += covMat1_off
	phase_offset2 = np.angle( (covMat2[2, 0] - covMat2[3, 1]) + 1j * (covMat2[3, 0] + covMat2[2, 1]))  # averaged phase offset
	logging.debug('residual phase shift 2 part 2:' + str(phase_offset2))
	res.covMat = covMat2 / yavg
	res.covMatOff = covMat2_off / yavg
	res.psi_mag_avg /= (xavg*yavg)
	dispData['select'] = selected
	res.n = np.zeros(2)                                         # photon number
	res.n[0] = (res.covMat[0, 0] + res.covMat[1, 1])                # Photon numbers
	res.n[1] = (res.covMat[2, 2] + res.covMat[3, 3])
	res.n += 0.5
	res.psi = (res.covMat[2, 0] - res.covMat[3, 1]) + 1j * (res.covMat[3, 0] + res.covMat[2, 1])
	res.c_avg = res.c_avg / (xavg*yavg)                 # Covariance matrix (FFTcov)
	res.psi_avg[0, :] = (res.c_avg[0] * 1.0 - res.c_avg[1] * 1.0 + 1j * (res.c_avg[2] * 1.0 + res.c_avg[3] * 1.0))
	res.sqph = np.angle(res.psi_avg[0][lags])               # residual Phase offset
	# Get Squeezing value, Ineq, and noise
	# res.sqph = phase_offset2
	res.sq, res.ineq, res.noise = get_sq_ineq(res.psi, res.n[0], res.n[1], np.float(dispData['f1']), np.float(dispData['f2']))
	logging.debug('n1full: ' + str(np.mean(on.I1 ** 2 + on.Q1 ** 2 - off.I1 ** 2 - off.Q1 ** 2)))
	logging.debug('On Matrix:' + str(res.covMat))


def get_data_avg(dispData, dicData):
	''' Procedure for phase corrections
	# 1. individual phase offsets in a list
	# 2.1 apply ind. phase offsets at each point
	# 2.2 average points and obtain second overall phase offset in a list
	# 3.1 use individual phase offsets + overall phase offsets
	# 3.2 average points in power
	What is returned?:
		res.n           # avg. photon numbers
		res.covMat      # avg. Covariance matrix with drive ON
		res.covMatOff   # avg. Covariance matrix with drive OFF
		res.psi         # average complex PSI
		res.sqph        # The residual phase offset
		res.sq          # Ammount of squeezing
		res.ineq        # calculated Ineq-Requirement
		res.noise       # lags noise (zeros as long as np.cov is used)
		res.IQmapMs_avg # The averaged 4x4 covariance matrix
	'''
	dd = dispData
	on, off, Fac1, Fac2 = prep_data(dd, dicData)
	res = dicData['res']
	lags = dd['lags']
	mapdim = dd['mapdim']
	startpos = dd['select']
	res.n = np.zeros(2)                                         # photon number
	res.IQmapM_avg = np.zeros([4, mapdim[0], mapdim[1]])        # histogram map
	res.c_avg = np.zeros([10, lags * 2 + 1])                    # Covariance Map inc PSI
	res.c_avg_off = np.zeros([10, lags * 2 + 1])                # Covariance Map
	res.psi_avg = 1j * np.zeros([1, lags * 2 + 1])              # PSI
	res.psi_mag_avg = np.zeros(2)
	covMat = np.zeros([4, 4])
	covMatOff = np.zeros([4, 4])
	for jj in range(dd['Power Averages']):
		dd['select'] = startpos + jj
		intermediate_covmatrix = np.zeros([10, lags * 2 + 1])   # Covariance Map inc PSI
		for i in range(dd['Averages']):
			assignRaw(dd, dicData)
			correctPhase(dd, dicData)                           # individual phase correction & assign on.CovMat
			intermediate_covmatrix += on.CovMat
			dd['select'] = dd['select'] + dd['dim1 pt']         # Jump to next section

	imed = intermediate_covmatrix
	residual_phase_offset = np.angle(imed[0] * 1.0 - imed[1] * 1.0 + 1j * (imed[2] * 1.0 + imed[3] * 1.0))[lags]
	logging.debug('Residual_phase_offset:' + str(residual_phase_offset))
	dd['select'] = startpos

	for jj in range(dd['Power Averages']):
		dd['select'] = startpos + jj
		for i in range(dd['Averages']):
			assignRaw(dd, dicData)
			logging.debug('Working on trace number ' + str(dd['Trace i, j, k']))
			logging.debug('dim1 value :' + str(dicData['dim1 lin'][int(dd['Trace i, j, k'][0])]))
			correctPhase(dd, dicData, avg_phase_offset=residual_phase_offset)  # assigns on.CovMat
			makehist2d(dd, dicData)
			res.IQmapM_avg += res.IQmapM
			res.c_avg += on.CovMat
			dd['select'] = dd['select'] + 201               # for now a hard coded number!
			covMat0 = np.cov([on.I1, on.Q1, on.I2, on.Q2])  # for now using numpies cov function to compare with FFT
			covMat1 = np.cov(
				[off.I1, off.Q1, off.I2, off.Q2])           # Turns out to be the same as using the FFT but seems slightly faster
			covMat += covMat0                               # w. drive ON
			covMatOff += covMat1                            # w. drive OFF
			covMat[0, 0] -= covMat1[0, 0]
			covMat[1, 1] -= covMat1[1, 1]
			covMat[2, 2] -= covMat1[2, 2]
			covMat[3, 3] -= covMat1[3, 3]
			res.psi_mag_avg[0] += np.abs(
				(covMat0[2, 0] - covMat0[3, 1]) + 1j * (covMat0[3, 0] + covMat0[2, 1]))  # magnitude w. drive ON
			res.psi_mag_avg[1] += np.abs(
				(covMat1[2, 0] - covMat1[3, 1]) + 1j * (covMat1[3, 0] + covMat1[2, 1]))  # magnitude w. drive OFF

	dd['select'] = startpos
	res.psi_mag_avg /= dd['Averages']
	res.n[0] = (covMat[0, 0] + covMat[1, 1])                # Photon numbers
	res.n[1] = (covMat[2, 2] + covMat[3, 3])
	res.n = 0.5 + res.n / dd['Averages']                    # force averaged value to be larger than 0.5
	res.covMat = covMat / dd['Averages']                    # Covariance matrix (np.cov)
	res.covMatOff = covMatOff / dd['Averages']
	res.psi = (res.covMat[2, 0] - res.covMat[3, 1]) + 1j * (res.covMat[3, 0] + res.covMat[2, 1])
	res.c_avg = res.c_avg / dd['Averages']                  # Covariance matrix (FFTcov)
	res.psi_avg[0, :] = (res.c_avg[0] * 1.0 - res.c_avg[1] * 1.0 + 1j * (res.c_avg[2] * 1.0 + res.c_avg[3] * 1.0))
	res.sqph = np.angle(res.psi_avg[0][lags])               # residual Phase offset
	# Get Squeezing value, Ineq, and noise
	res.sq, res.ineq, res.noise = get_sq_ineq(res.psi, res.n[0], res.n[1], np.float(dd['f1']), np.float(dd['f2']))
	logging.debug('n1full: ' + str(np.mean(on.I1 ** 2 + on.Q1 ** 2 - off.I1 ** 2 - off.Q1 ** 2)))
	logging.debug('On Matrix:' + str(res.covMat))



def get_sq_ineq(psi, n1, n2, f1, f2):
	'''returns the ammount of squeezing, ineq and noise'''
	noise = np.sqrt(np.var(np.abs(psi)))
	logging.debug('Mag Psi sqrt(Variance): ' + str(noise))
	squeezing = np.max(np.abs(psi)) / ((n1 + n2) / 2.0)
	logging.debug(('n1: ' + str(n1) + ' n2: ' + str(n2)))
	a = 2.0 * np.sqrt(f1 * f2) * np.abs(n1 + n2 - 1)
	b = f1 * (2.0 * n1 + 1.0 - 0.5) + f2 * (2.0 * n2 + 1.0 - 0.5)
	ineq = a / b  # does not include zpf
	logging.debug(('ineq: ' + str(ineq) + ' sq raw: ' + str(squeezing)))
	return squeezing, ineq, noise
