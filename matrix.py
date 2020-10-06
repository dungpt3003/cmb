#!/usr/bin/env python
# This files contains functions to construct matrices 
import numpy as np

def loadMatrix(aFile):
    with open(aFile, "r") as f:
        A = np.loadtxt(f, delimiter=",")
        return A

def loadDiagonalMatrix(aFile):
    with open(aFile, "r") as f:
        temp = np.loadtxt(f,delimiter=",")
        T = np.zeros((temp.shape[0], temp.shape[0]))
        for i in range(0,temp.shape[0]):
            T[i,i] = temp[i]
        return T

if __name__ == '__main__':
    A = loadMatrix("data/a.csv")
    T = loadDiagonalMatrix("data/t.csv")
    print(A)
    print(T)