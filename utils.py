#!/usr/bin/env python
from scipy import constants
import numpy as np

V_PLANCK = [x * (10**9) for x in [30.0, 44.0, 70.0, 100.0, 143.0, 217.0, 353.0, 545.0, 857.0]]
V_0 = V_PLANCK[4]
PLANCK_H = constants.Planck
BOLTZMANN_K = constants.Boltzmann
K_S = -2.65 
K_D = 1.5
K_FF = -2.14
T1 = 18.1

# Calculate conversion factor c(v) based on the formula:
# c(v) = (e^psi − 1)^2/psi^2e^psi where psi = hv/k_BT1
def calc_conversion_factor(v):
    psi = (PLANCK_H) * v / (BOLTZMANN_K * T1)
    e_psi = np.exp(psi)
    return ((e_psi - 1) ** 2) / (psi*psi * e_psi)

# Calculate B(v) based on the formula:
# B(v) = v/[exp(hv/kBT1) − 1]
def B(v):
    return v / (np.exp(PLANCK_H/(BOLTZMANN_K * T1)) - 1)

# Calculate synchrotron value based on the formula:
# a_s(v, v0) = c(v) * (v/ v0) ^ k_s 
def calc_synchrotron(v):
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
def calculateA(m, n):
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

A = pretty_print_matrix(calculateA(9, 4))