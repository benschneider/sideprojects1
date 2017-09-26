#! /Library/Frameworks/Python.framework/Versions/3.6/bin/python3 -i
from scipy.optimize import curve_fit, minimize
import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from struct import pack, unpack

plt.ion()


def savemtx(filename, data, header='Units,ufo,d1,0,1,d2,0,1,d3,0,1'):
    with open(filename, 'wb') as f:
        f.write(bytes(header + '\n', 'utf-8'))
        mtxshape = data.shape
        line = str(mtxshape[2]) + ' ' + str(mtxshape[1]) + ' ' + str(mtxshape[0]) + ' ' + '8'
        f.write(bytes(line + '\n', 'utf-8'))  # 'x y z 8 \n'
        for ii in range(mtxshape[2]):
            for jj in range(mtxshape[1]):
                content = pack('%sd' % mtxshape[0], *data[:, jj, ii])
                f.write(content)
        f.close()


def load_data(fileName):
    a = open(fileName, 'r+')
    b = a.readlines()
    data = np.zeros([3, len(b[:-3])])
    for i, e in enumerate(b[:-3]):
        strings = e.split(' ')
        data[0, i] = (float(strings[0]))  # raw xaxis
        data[2, i] = (float(strings[1]))  # raw yaxis
    return data


# def return_2p_value(data, xval):
#     # find closest value and approximate intermediate value
#     xlength = len(data[1])
#     xstep = (data[1][-1] - data[1][0]) / (xlength-1)
#     idx = (xval-data[1][0])/xstep
#     int_idx = int(idx)
#     resid_idx = idx - int_idx
#     resid = (data[2][int_idx+1] - data[2][int_idx]) * resid_idx
#     return data[2][int_idx] + resid


def get_s11(flux, Ic, Cap, R=1e99):
    flux0 = 2.07e-15
    f0 = 4.1e9
    z0 = 50
    L = flux0 / (Ic * 2.0 * pi * np.abs(np.cos(pi * flux / flux0)) + 1e-90)
    Ysq = (1.0 / R + 1.0 / (1j * 2.0 * pi * f0 * L + 1j * 1e-90) + 1j * 2.0 * pi * f0 * Cap)
    zsq = 1.0 / Ysq
    s11 = (zsq - z0) / (zsq + z0)
    return s11


def fitFunc_mag(flux, Ic, Cap, R, offset2, scales2):
    s11 = get_s11(flux, Ic, Cap, R)
    return scales2 * np.abs(s11) + offset2


def fitFunc_ang(flux, Ic, Cap, offset1=0.0, slope=0.0):
    s11 = get_s11(flux, Ic, Cap)
    return np.angle(s11) + offset1 - slope * flux / flux0


def parabola(a, f, x):
    if a < 1e-5:
        return x * 0.0
    elif f < 1e-20:
        return x * 0.0
    elif f > 87.6e9:  # no effects above gap
        return x * 0.0
    else:
        b = f / 2.0
        z = (-a / b**2 * (x - b)**2 + a)
        z[z < 0] = 0
    return z


def get_full_parabola(freqaxis, fft_drive, scale=0.01):
    parabola_full = np.zeros_like(freqaxis)
    for i, amp in enumerate(fft_drive):
        freq = freqaxis[i]
        if freq > gap_freq:
            break
        parabola_full += parabola(amp * scale, freq, freqaxis)
    return parabola_full


def get_drive(amplitude, dc_offset, omega0, timeaxis):
    drive = np.zeros([4, resolution])
    for i, jj in enumerate(timeaxis):
        signal = amplitude * np.sin(jj * omega0) + dc_offset
        drive[0, i] = signal
        drive[1, i] = fitFunc_mag(signal * flux0, popt2[0], popt2[1], popt2[2], popt2[3], popt2[4])
        drive[3, i] = fitFunc_ang(signal * flux0, Ic, Cap, offset, slope)
    return drive


def get_fft_responses(drive):
    loss = np.mean(drive[1])
    m0 = np.min(drive[0]) + (np.max(drive[0]) - np.min(drive[0])) / 2.0
    m3 = np.min(drive[3]) + (np.max(drive[3]) - np.min(drive[3])) / 2.0
    fft_signal = np.real(np.fft.rfft(drive[0] - m0, norm="ortho"))
    fft_mirror = np.real(np.fft.rfft(drive[3] - m3, norm="ortho")) * loss
    # fft_loss = np.abs(np.fft.rfft(drive[1], norm="ortho"))
    return fft_signal, fft_mirror

fileName_ang = '1157_S11_f2_D2vAvg_phase.l.0.linecut.dat'
fileName_mag = '1157_S11_f2_D2vAvg_mag.l.0.linecut.dat'
output_file = 'parabolas_all_lp4_interference.mtx'
data = load_data(fileName_ang)
data_mag = load_data(fileName_mag)
x0 = -0.792
x1 = 4.03
data[1] = ((data[0] - x0) / (x1 - x0)) - 0.5  # corrected xaxis
# Fitting Squid Phase response
R = 300
Ic = 2.4e-6
Cap = 3e-13
flux0 = 2.07e-15    # Tm^2; Flux quanta: flux0 =  h / (2*charging energy)
flux = np.linspace(-0.75, 0.7011, 701) * flux0
offset = -4.5
slope = 0.1
offset2 = 0.0
scale2 = 1.0
iguess = [Ic, Cap, offset, slope]
iguess2 = [Ic, Cap, R, offset2, scale2]
# ub = [1e-4, 1e-9, 10, 10]
# lb = [1e-9, 1e-30, -10, -10]
# bounds1 = (lb , ub)
# ub2 = [1e-4, 1e-9, 1e30, 2, 2]
# lb2 = [1e-9, 1e-30, 1, -2, -2]
# bounds2 = (lb2, ub2)
popt, pcov = curve_fit(fitFunc_ang, flux, data[2], p0=iguess)
data_mag[2] = data_mag[2] / np.max(data_mag[2])
popt2, pcov2 = curve_fit(fitFunc_mag, flux, data_mag[2], p0=iguess2, maxfev=500000, xtol=1e-36)
print(popt)
print(pcov)
Ic = popt[0]
Cap = popt[1]
offset = popt[2]
slope = popt[3]
print(popt2)
print(pcov2)
resolution = 2**12
pumpfreq = 8.9e9
omega0 = 2.0 * np.pi * pumpfreq
timeaxis = np.linspace(0, 20e-9, resolution)
freqaxis = np.fft.rfftfreq(timeaxis.shape[-1], (timeaxis[1] - timeaxis[0]))
gap_freq = 88e9
pump_idx = np.argmin(abs(freqaxis - pumpfreq))
scale = 0.01
powerpoints = 361
fluxpoints = 401
dc_offsets = np.linspace(-0.8, 1.2, fluxpoints)
amplitudes = np.linspace(0.0, 0.60, powerpoints)
parabolas = np.zeros([fluxpoints, powerpoints, len(freqaxis)])
parabolas2 = np.zeros([fluxpoints, powerpoints, len(freqaxis)])

for kk, dc_offset in enumerate(dc_offsets):
    print(kk)
    for jj, amplitude in enumerate(amplitudes):
        drive = get_drive(amplitude, dc_offset, omega0, timeaxis)
        fft_signal, fft_mirror = get_fft_responses(drive)
        amp_d = fft_mirror[pump_idx]
        parabolas[kk, jj] = get_full_parabola(freqaxis, fft_mirror, scale)
        # parabolas[1, jj] = parabola(amp_d * scale, pumpfreq, frepqaxis)
        # parabolas[2, jj] = (parabolas[1, jj] + 1e-199) / (parabolas[0, jj] + 1e-90)
        # parabolas[3, jj] = parabolas[1, jj] - parabolas[0, jj]

header = ('Units,ufo,Frequency,' + str(freqaxis[0]) + ',' + str(freqaxis[-1]) +
          ',Pump,' + str(amplitudes[0]) + ',' + str(amplitudes[-1]) +
          ',d3,-0.5,1.0')
savemtx(output_file, parabolas, header)

m0 = np.min(drive[0]) + (np.max(drive[0]) - np.min(drive[0])) / 2.0
m3 = np.min(drive[3]) + (np.max(drive[3]) - np.min(drive[3])) / 2.0
plt.figure(1)
plt.plot(flux / flux0, data_mag[2])
plt.plot(flux / flux0, fitFunc_mag(flux, popt2[0], popt2[1], popt2[2], popt2[3], popt2[4]))
plt.figure(2)
plt.plot(flux / flux0, data[2])
plt.plot(flux / flux0, fitFunc_ang(flux, popt[0], popt[1], popt[2], popt[3]))
plt.figure(4)
plt.plot(timeaxis, drive[0] - m0)
plt.figure(5)
plt.plot(timeaxis, drive[3] - m3)
plt.figure(6)
plt.plot(freqaxis, fft_signal)
plt.figure(7)
plt.plot(freqaxis, fft_mirror)
plt.figure(8)
plt.plot(freqaxis, parabolas[0, jj])
