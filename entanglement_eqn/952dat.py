import parsers as p
import numpy as np

Logn = p.loaddat('952_G50mV_LogN.c.208.linecut.dat')
LognMax = p.loaddat('952_G50mV_LogN_max.c.208.linecut.dat')
LognMin = p.loaddat('952_G50mV_LogN_min.c.208.linecut.dat')

SqCon = p.loaddat('952_G50mV_ineqSq.c.208.linecut.dat')
SqConMax = p.loaddat('952_G50mV_ineqSq_max.c.208.linecut.dat')
SqConMin = p.loaddat('952_G50mV_ineqSq_min.c.208.linecut.dat')


x = np.array(Logn[0])
Neg = np.array(Logn[1])
NegMax = np.array(LognMax[1])
NegMin = np.array(LognMin[1])
Sq = np.array(SqCon[1])
SqMax = np.array(SqConMax[1])
SqMin = np.array(SqConMin[1])

data = np.zeros([7, len(x)])
data[0, :] = x
data[1, :] = Neg
data[2, :] = NegMin
data[3, :] = NegMax
data[4, :] = Sq
data[5, :] = SqMin
data[6, :] = SqMax

p.savedat('952.dat', data)
