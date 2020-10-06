#!/usr/bin/env python
# This files contains functions to construct matrices 
import numpy as np

def loadMatrix(aFile):
    with open(aFile, "r") as f:
        A = np.loadtxt(f, delimiter=",")
        return A

def loadDigonalMatrix(aFile):
    with open(aFile, "r") as f:
        temp = np.loadtxt(f,delimiter=",")
        T = np.zeros((self.num_m,self.num_m))
        for i in range(0,self.num_m):
            T[i,i] = temp[i]
        return T
