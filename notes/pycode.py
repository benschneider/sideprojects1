from scipy.constants import Boltzmann as Kb
from scipy.constants import h, e, pi

h*e*pi*Kb

# xoffsets:
# with xoff = pos of 0.5 flux quanta
# dx = corres distance of flux quanta
xoff = 139.3e-3
x1flux = 479.6e-3
x = (x-xoff)/x1flux + 0.5
# (x - 139.3e-3)/479.6e-3+0.5
