#!/usr/bin/env python
from scipy import constants
import numpy as np
import math

V_PLANCK = [x * (10**9) for x in [30.0, 44.0, 70.0, 100.0, 143.0, 217.0, 353.0, 545.0, 857.0]]
V_0 = V_PLANCK[4]
PLANCK_H = constants.Planck
BOLTZMANN_K = constants.Boltzmann
K_S = -2.65 
K_D = 1.5
K_FF = -2.14
T1 = 18.1
L = "left"
R = "right"
U = "up"
D = "down"

# Calculate conversion factor c(v) based on the formula:
# c(v) = (e^psi − 1)^2/psi^2e^psi where psi = hv/k_BT1
def calc_conversion_factor(v):
    psi = (PLANCK_H) * v / (BOLTZMANN_K * T1)
    e_psi = np.exp(psi)
    print(psi)
    return ((e_psi - 1) ** 2) / (psi*psi * e_psi)

# Calculate B(v) based on the formula:
# B(v) = v/[exp(hv/kBT1) − 1]
def B(v):
    return v / (np.exp(PLANCK_H/(BOLTZMANN_K * T1)) - 1)

# Calculate synchrotron value based on the formula:
# a_s(v, v0) = c(v) * (v/ v0) ^ k_s 
def calc_synchrotron(v):
    print(calc_conversion_factor(v))
    return calc_conversion_factor(v) * ((v / V_0) ** K_S)

# Calculate galatic dust value based on the formula:
# a_d(v, v0) = c(v) * B(v) / B(v0) * (v / v0)^k_d
def calc_galatic_dust(v):
    return calc_conversion_factor(v) * B(v) / B(V_0) * ((v / V_0)**K_D)

# Calculate free-free emission value based on the formula:
# a_ff(v, v0) = c(v)* (v/ v0)^k_ff
def calc_free_emission(v):
    return calc_conversion_factor(v) * ((v / V_0) ** K_FF)

# Calculate the mxn matrix A
# The first column is all 1 (CMB source)
# The n-1 other columns represent other sources: synchrotron, glatic dust, free-free emission
def calculate_matrix_A(m, n):
    A = np.zeros((m, n))
    A[:, 0] = 1
    A[:, 1] = [calc_synchrotron(x) for x in V_PLANCK]
    A[:, 2] = [calc_galatic_dust(x) for x in V_PLANCK]
    A[:, 3] = [calc_free_emission(x) for x in V_PLANCK]
    return A

def pretty_print_matrix(X):
    m = X.shape[0]
    n = X.shape[1]
    for i in range(m):
        for j in range(n-1):
            print(X[i, j], end=',')
        print(X[i, n-1])
    return

def find_neighbors(i, m, n):
    """find all the neighbours of i
    Parameters
    ----------
    i : int
        the index of the point that we need to find its neighbours
    m : int
        the number of rows of original matrix (3*Nside)
    n : int
        the number of columns of original matrix (4*Nside)
    """

    # All the neighours of i
    res = {
            L: i - m,
            R: i + m,
            U: i - 1,
            D: i + 1
        }
        
    # Remove all the invalid neighbours
    if i % m == 0: # The first row
        res.pop(U)
    elif (i % n) == (n - 1): #The last row
        res.pop(D)
    if i - m < 0: # The first column
        res.pop(L)
    elif i + m > n * m - 1: # The last column
        res.pop(R)
    return res.values()

def calc_D_size(v):
    # sqrt = int(math.sqrt(v))
    # for i in range(sqrt, 0, -1):
    #     if v % i == 0:
    #         return (i, v/i)
    base = 2**v
    return 3*base, 4*base

def calculate_N_from_level(lvl):
    # N = 12 * N_side ^ 2 = 12 * (2*lvl)^2
    return (4**lvl) * 12
