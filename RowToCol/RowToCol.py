#!/usr/bin/python
import numpy as np
import sys

''' This script load an asci file whose data is set to rows and rotates it
such that the output file is in colums. Essentially transposes the data.'''


def abort1():
    sys.exit("usage is: RowToCol.py filein fileout")
try:
    fnameout = sys.argv[-1]
    fname = sys.argv[-2]
    if fnameout == 'RowToCol.py' or fname == 'RowToCol.py':
        abort1()
except:
    abort1()

dataIn = np.loadtxt(fname)
dataout = dataIn.transpose()
np.savetxt(fnameout, dataout, delimiter='\t')
