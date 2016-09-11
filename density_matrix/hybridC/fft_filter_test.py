from scipy.signal.signaltools import _next_regular
from matplotlib import pyplot as plt
from numpy.fft import fft, rfftn, irfftn, fftshift # for real data can take advantage of symmetries
import numpy as np
import codecs, json
# from scipy.signal import remez, freqz, lfilter
# lpf = remez(21, [0, 0.2, 0.3, 0.5], [1.0, 0.0])
# w, h = freqz(lpf)
#
# t = np.arange(0, 1.0, 1.00/1000)
# # s = np.sin(2*np.pi*100*t) + np.sin(2*np.pi*200*t)
# noise_amp = 5.0
# s = np.sin(2*np.pi*100*t) + np.sin(2*np.pi*200*t)  # +noise_amp * np.random.randn(len(t))
#
# # sout = lfilter(lpf, 1, s)
# # plt.figure(1)
# # plt.plot(s[:100])
# # plt.plot(sout[:100])
#
# ft = fftshift(fft(s)/len(s))
# # ft2 = np.fft.fft(sout[40:])/len(sout)
# # plt.plot(20.0*np.log10(np.abs(ft2)))
# # # plt.plot((np.abs(ft)))
# # plt.show()
#
# shap0 = np.array(s.shape) - 1
# fshape = [_next_regular(int(d)) for d in shap0]  # padding to optimal size for FFTPACK
# ft11 = fftshift(rfftn(s, fshape)/fshape)
#
# plt.figure(3)
# # plt.plot(w/(2*np.pi), abs(h))
# # plt.plot(20.0*np.log10(np.abs(ft11)))
# plt.plot(np.abs(ft11))
#
# plt.figure(4)
# #plt.plot(20.0*np.log10(np.abs(ft)))
# plt.plot(np.abs(ft))
# plt.show()
# a = np.random.rand(5)
# b = np.random.rand(5) + 0.1*a


# a = np.arange(10).reshape(2,5) # a 2 by 5 array
# b = a.tolist() # nested lists with same data, indices
# file_path = "blub.json" ## your path variable
# def save_json(file_path, stuff):
#     json.dump(stuff, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
#
# def load_json(file_path):
#     obj_text =  codecs.open(file_path, 'r', encoding='utf-8').read()
#     return json.loads(obj_text)
