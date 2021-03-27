#!/usr/bin/env python

import time
import numpy as np
import sys
from matrix import *
from utils import *
from conjugate_solution import ConjugateSolution


def generate_random_sources(h):
    base_patch = 4 ** h
    S = np.random.rand(4, base_patch)
    return S

def generate_sources(m, n, h):
    repetition = 12
    base_patch = 4 ** h
    N = repetition * base_patch
    A = calculate_matrix_A(m, n)
    y = np.zeros(0)
    S_ = np.zeros(0)
    mean = np.zeros(m)
    T = load_diagonal_matrix_inverse("data/t.csv")
    print("Mean: ", mean)
    print("Covariance: ", T)
    for i in range(repetition):
        S = generate_random_sources(h)
        S_ = np.append(S_, S)
        print(S)
        for j in range(base_patch):
            y_temp = np.matmul(A, S[:, j]) + np.random.multivariate_normal(mean, T)
            y = np.append(y, y_temp)
    return y, S_

    
if __name__ == '__main__':
    m = 9
    n = 4
    sourcefilea = "data/a.csv"
    sourcefilet = "data/t.csv"
    tnum = 4

    for i in range(1,2):
        
        print("-------------lvl:{0}, N:{1}------------".format(i, calculate_N_from_level(i)))
        #y = 5 * np.random.randint(0,10,(m * calculate_N_from_level(i), 1))
        #iterate method
        y, S = generate_sources(9, 4, i)
        start = time.time()
        ite_s = ConjugateSolution(sourcefilea,sourcefilet,m,n,i,y,0.001,tnum)
        ite_x = ite_s.findSolution()
        end = time.time()
        print("iterate method took:{0}s".format(end - start))
        print("Solution:", ite_x)
        print("Original", S)
        sys.stdout.flush()

        #std solution
        #start = time.time()
        #std_s = StdSolu(sourcefilea,sourcefilet,m,n,i,y)
        #std_x = std_s.findSolution()
        #end = time.time()
        #print("standard method took:{0}s".format(end - start))
        #sys.stdout.flush()

        #distance = std_x - ite_x
        #print("distance square:", np.dot(distance,distance))
        #sys.stdout.flush()
