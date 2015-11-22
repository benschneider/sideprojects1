import numpy as np

f1 = 4.1e9
f2 = 4.8e9
n = 0.1


def sqeezing_req(f1, f2, n):
    ''' This function calculates the amount of squeezing required
    to prove entanglement
    '''
    return ((2.0*np.sqrt(f1*f2)*(2*n))
            / (f1*(2*n+1)+f2*(2*n+1)))


# Covaricance Matrix derived from measurement results

M180mV = np.array([[0.255, 0.0, 0.01, 0.0095],
                   [0.0, 0.255, 0.067,  -0.086],
                   [0.01, 0.067, 0.255, 0.0],
                   [0.0095, -0.086, 0.0, 0.255]])

M200mV = np.array([[0.26, 0.0, 0.01, 0.0102],
                   [0.0, 0.26, 0.01, -0.014],
                   [0.01, 0.01, 0.26, 0.0],
                   [0.0102, -0.014, 0.0, 0.26]])

M220mV = np.array([[0.3, 0.0, 0.028, 0.012],
                   [0.0, 0.3, 0.012, -0.03],
                   [0.028, 0.012, 0.3, 0.0],
                   [0.012, -0.03, 0.0, 0.3]])

# Ntest is an identity matrix to see how many noise photons would
# be required before the Negativity value becomes zero thus classical.
# so far it appears even in the worst case scenario about 0.5
# additional noise photons would be required to make out entanglement claim
# invalid. This is quite good! as our error bars are indeed much smaler!
Ntest = np.array([[0.30, 0.0, 0.0, 0.0],
                  [0.0, 0.30, 0.0, 0.0],
                  [0.0, 0.0, 0.30, 0.0],
                  [0.0, 0.0, 0.0, 0.30]])


# function to calculate LogNegativity from the Matrix
def get_LogNegNum(CovM):
    V = np.linalg.det(CovM)
    A = np.linalg.det(CovM[:2, :2])
    B = np.linalg.det(CovM[2:, 2:])
    C = np.linalg.det(CovM[:2, 2:])
    s = A + B - 2.0*C
    vn = np.sqrt(s/2.0 - np.sqrt(s**2.0 - 4.0*V) / 2)
    return -np.log10(2.0*vn)

M180mV_n = get_LogNegNum(M180mV)
M200mV_n = get_LogNegNum(M200mV)
M220mV_n = get_LogNegNum(M220mV)

test_n = get_LogNegNum(M180mV + Ntest)
