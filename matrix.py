#!/usr/bin/env python
# This files contains functions to construct matrices 
import numpy as np

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


if __name__ == '__main__':
    A = loadMatrix("data/a.csv")
    T = loadDiagonalMatrix("data/t.csv")
    print(A)
    print(T)