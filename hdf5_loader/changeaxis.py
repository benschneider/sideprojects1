import numpy as np
from scipy.interpolate import griddata


def interp_y(y, zmat, factor=10):
    ''' given x axis and Zmatrix
        interpolates missing data along the x-axis
        which is remapped on a new x-axis which is 'factor' times larger
        then return the new bigger matrix including the missing
        interpolated data

    '''
    y2 = np.linspace(y[0], y[-1], len(y)*factor)
    resultmatrix = np.zeros(len(y2))
    for i in range(zmat.shape[1]):
        newline = griddata(y, zmat[:, i], y2, method='linear')
        resultmatrix = np.dstack((resultmatrix, newline))

    return np.array(resultmatrix[0, :, :-1])
    # return np.array(y2), np.array(resultmatrix[0, :, :-1])
