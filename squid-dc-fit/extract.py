# import matplotlib
# matplotlib.use('macosx') # macosx
# import matplotlib.pyplot as plt
import bottleneck as bn
import numpy as np
import Gnuplot as gp
from parsers import loadmtx, read_header_old
# , make_header, dim, savemtx, savedat
from scipy.constants import h, e, pi

# mtx file to be loaded
# filename1 = "data/S1_164_voltage_adj.mtx"
# filename1 = "data/S1_160_voltage_adj2.mtx"
# filename1 = "data/S1_420_voltage_adj.mtx"
filename1 = "data/S1_905_SI.mtx"
data, head = loadmtx(filename1)
d1, d2, d3, dz = read_header_old(head, Data=data)
flux0 = h/(2.0*e)

# manual found flux offset
# d2.scale = 0.5/0.237
# d2.update_lin()
# d2.off = 0.177*d2.scale
# d2.lin = d2.lin+d2.off

# d2.scale = 1.0/0.85
# d2.update_lin()
# d2.lin = d2.lin-0.207


def xderiv(d2MAT, d1):
    # this calculates a 3p derivative
    # prepare 3 arrays for deriv calculations
    a2 = np.zeros([d2MAT.shape[0], d2MAT.shape[1]+2])
    a2[:, 1:-1] = data[0]
    m1 = data[0] - a2[:, :-2]
    m2 = a2[:, 2:] - data[0]
    dy = (m1+m2)/2.0
    # approximate edge points
    dy[:, 0] = dy[:, 1]
    dy[:, -1] = dy[:, -2]
    dx = d1.lin[1]-d1.lin[0]
    return dy/dx

d1mat = xderiv(data[0], d1)
resultmat = np.zeros([d2.pt, 8])
line = np.zeros(d1.pt)
fjosephson = 340e-6 * 483.6e6/1e-6


def find_nearest(someArray, value):
    '''
    Returns an index number (idx)
    at which the someArray.[idx] is closest to value
    '''
    idx = abs(someArray - value).argmin()
    return idx


def extractIrIcR(line, d1):
    # zeroIdx assumes that there is no offset in the axis
    zeroIdx = round(d1.start*d1.pt/(d1.start-d1.stop))
    zeroIdx = 452  # Temp override the autofound value
    l1 = line[:zeroIdx]
    l2 = line[zeroIdx:]
    IrIdx = np.sort(bn.argpartsort(-l1, 3)[:3])
    IcIdx = np.sort(bn.argpartsort(-l2, 13)[:13]) + zeroIdx
    # store only peaks closest to zero
    Ir = d1.lin[IrIdx[-1]]*1e-6
    Ic = d1.lin[IcIdx[1]]*1e-6
    R = np.mean(line[IrIdx[:]])
    return Ir, Ic, R


for ii in range(d2.pt):
    line = np.array(d1mat[ii])
    Ir, Ic, R = extractIrIcR(line, d1)
    # R = 50
    # if Ir < 1e-20:
    #     Ir = 1e-20

    betac = ((4.0*Ic)/(pi*(Ir)))**2.0
    R2C = betac * flux0/(2.0*pi*Ic)
    flux = d2.lin[ii]
    L = flux0 / (Ic*2.0*pi * np.abs(np.cos(pi*flux)) + 1e-9)
    # C = 70e-15
    C = (R2C/(R**2))
    # R = np.sqrt(R2C/C)
    plasmaf = 1.0/(2*pi*np.sqrt(L*C))
    resultmat[ii] = [d2.lin[ii], Ir, Ic, R, C, L, plasmaf, betac]


a = np.array(d1mat[0])
l1 = np.zeros([d1.pt, 2])
l1[:, 0] = d1.lin
l1[:, 1] = d1mat[0]

# np.savetxt('test.dat', l1, delimiter='\t')
np.savetxt('test2.dat', resultmat, delimiter='\t')


def prepFig(g1):
    g1("set xrange [" + str(d2.lin[0]) + ":" + str(d2.lin[-1]) + "]")
    g1("set key top right")
    g1.xlabel("Flux ({/Symbol \106 /\106_0})")
    g1("set log y")
    g1("set term x11 0")
    g1("set autoscale y")

g1 = gp.Gnuplot(persist=0, debug=1)
prepFig(g1)
g1("plot 'test2.dat' u 1:(-$2*1e6) w l t 'Ir (uA)'")

g2 = gp.Gnuplot(persist=0, debug=1)
prepFig(g2)
g2("plot 'test2.dat' u 1:($3*1e6) w l t 'Ic (uA)'")

g3 = gp.Gnuplot(persist=0, debug=1)
prepFig(g3)
g3("plot 'test2.dat' u 1:($4/1000) w l t 'R (kOhm)'")

g4 = gp.Gnuplot(persist=0, debug=1)
prepFig(g4)
g4("unset log y")
g4("set yrange [39.5:40.5]")
g4("plot 'test2.dat' u 1:($5*1e15) w l t 'Cap (fF)'")

g5 = gp.Gnuplot(persist=0, debug=1)
prepFig(g5)
g5("plot 'test2.dat' u 1:($6*1e9) w l t 'L (nH)'")

g6 = gp.Gnuplot(persist=0, debug=1)
prepFig(g6)
g6("plot 'test2.dat' u 1:($7/1e9) w l t 'f_P (GHz)'")
g6("replot 8")

g7 = gp.Gnuplot(persist=0, debug=1)
prepFig(g7)
g7("plot 'test2.dat' u 1:($8) w l t 'BetaC'")

g8 = gp.Gnuplot(persist=0, debug=1)
prepFig(g8)
g8("plot 'test2.dat' u 1:($8) w l t 'BetaC'")
g8("replot 'test2.dat' u 1:($7/1e9) w l t 'f_P (GHz)'")
g8("replot 8")
g8("replot 'test2.dat' u 1:(-$2*1e6) w l t 'Ir (uA)'")
g8("replot 'test2.dat' u 1:($3*1e6) w l t 'Ic (uA)'")
g8("replot 'test2.dat' u 1:($4/1000) w l t 'R (kOhm)'")
g8("replot 'test2.dat' u 1:($5*1e15) w l t 'Cap (fF)'")
# g8("replot 'test2.dat' u 1:($6*1e9) w l t 'L (nH)'")

g8("file = 'data/output.eps'")
g8("load 'print.gnu'")
