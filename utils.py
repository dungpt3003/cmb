#!/usr/bin/env python
from scipy import constants

V_PLANCK = [30.0, 44.0, 70.0, 100.0, 143.0, 217.0, 353.0, 545.0, 857.0]
V_0 = 143.0
PLANCK_H = constants.Planck
BOLTZMANN_K = constants.Boltzmann
K_S = -2.65 
K_D = 1.5
K_FF = -2.14
T1 = 18.1

# Calculate conversion factor c(v) based on the formula:
# c(v) = (e^psi − 1)^2/psi^2e^psi where psi = hv/k_BT1
def calc_conversion_factor(v):
    psi = (PLANCK_H) * v / float(BOLTZMANN_K * T1)
    e_psi = np.exp(psi)
    return ((e_psi - 1) ** 2) / (psi*psi * e_psi))

# Calculate synchrotron value based on the formula:
# a_s(v, v0) = c(v) * (v/ v0) ^ k_s 
def calc_synchrotron(v):
    return calc_conversion_factor(v) * ((v / V_0) ** K_S)


