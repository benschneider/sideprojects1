import numpy as np


def xderiv(d2MAT, dx=1.0):
    '''
    This  derivative is inaccurate as the edges.
    Calculates a 3p derivative of a 2D matrix.
    This does not require you to shift the xaxis by one half pt.
    dx = distance between points
    '''
    a2 = np.zeros([d2MAT.shape[0], d2MAT.shape[1]+2])
    a2[:, 1:-1] = d2MAT
    m1 = d2MAT - a2[:, :-2]
    m2 = a2[:, 2:] - d2MAT
    dy = (m1+m2)/2.0
    # approximate edge points
    dy[:, 0] = dy[:, 1]
    dy[:, -1] = dy[:, -2]
    return dy/dx
