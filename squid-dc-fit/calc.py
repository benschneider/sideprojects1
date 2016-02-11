# import numpy as np
# import gnuplotlib as gp
from scipy.constants import h, e, pi

flux0 = h/(2.0*e)
Ic = 3.31e-6
Ir = 0.05e-6
betac = ((4.0*Ic)/(pi*Ir))**2.0
R2C = betac * flux0/(2.0*pi*Ic)

# print str(R2C/(4600.0**2)) + " F"
# x = np.arange(101) - 50
# gp.plot(x**2)
