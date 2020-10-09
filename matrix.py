#!/usr/bin/env python
# This files contains functions to construct matrices 
import numpy as np
from utils import calculateA

# Load normal matrix from file
def loadMatrix(aFile):
    with open(aFile, "r") as f:
        A = np.loadtxt(f, delimiter=",")
        return A

# Load values from file and create a diagonal matrix
def loadDiagonalMatrix(aFile):
    with open(aFile, "r") as f:
        temp = np.loadtxt(f,delimiter=",")
        T = np.zeros((temp.shape[0], temp.shape[0]))
        for i in range(0,temp.shape[0]):
            T[i,i] = temp[i]
        return T

# Generate matrix matrix B as the Kronecker product of A and the identity matrix size N
def generateMatrixB(A, N):
    return np.kron(A, np.eye(N))

# Generate matrix C as the Kronecker product of two diagonal matrices (T and diagN)
def generateMatrixC(T, N):
    diagN = np.eye(N)
    return np.kron(T, diagN)

def generateD(lvl):
    """ compute matrix D
    Parameters:
    ----------
    lvl : int
        Power Level
    """
    size = calNFromLvl(lvl)
    (m,n) = mostSqure(size)
    res = np.zeros((size,size),dtype=int)

    for i in range(0, size):
        # fill all the neighbours of i
        diag = 0
        for pos in findNeighbors(i,m,n):
            res[i][pos] = 1
            diag += 1
        res[i][i] = -1*diag
    return res


def calNFromLvl(lvl):
    return 4**lvl*12