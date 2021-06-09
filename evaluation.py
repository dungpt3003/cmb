#!/usr/bin/env python

import time
import numpy as np
import sys
from matrix import *
from utils import *
from conjugate_solution import ConjugateSolution

def generate_random_sources(n, m, h, mean_s, sd_s, printable=True):
    """Generate random source s and source vector y
    Parameters
    ----------
    n : int
        number of frequencies
    m : int
        number of diffuse sources
    h : int
        HEALPix level
    mean_s: float
        Mean value of the random source s
    sd_s: float
        Standard deviation value of the random source s
    """
    A = calculate_matrix_A(n, m)
    N = calculate_N_from_level(h)
    B = generate_matrix_B(A, N)
    T = load_diagonal_matrix("data/t.csv")
    C = generate_matrix_C(T, N)
    s = np.random.normal(mean_s, sd_s, m*N)

    mean_y = np.matmul(B, s)
    sd = np.array([1/np.sqrt(x) for x in np.diag(C)])
    y = np.random.normal(mean_y, sd, mean_y.shape[0])
    if printable:
        np.savetxt('output/random_y.txt', [y], delimiter=',', fmt='%0.20f')
        np.savetxt('output/random_s.txt', [s], delimiter=',', fmt='%0.20f')

    return s, y
    
if __name__ == '__main__':
    m = 9
    n = 4
    sourcefilea = "data/a.csv"
    sourcefilet = "data/t.csv"
    tnum = 4

    generate_random_sources(9, 4, 3, 1, 0.1, True)
    # for i in range(1,10):
        
    #     print("-------------lvl:{0}, N:{1}------------".format(i, calculate_N_from_level(i)))
    #     #y = 5 * np.random.randint(0,10,(m * calculate_N_from_level(i), 1))
    #     #iterate method
    #     y, S = generate_sources(9, 4, i)
    #     start = time.time()
    #     ite_s = ConjugateSolution(sourcefilea,sourcefilet,m,n,i,y,0.001,tnum)
    #     ite_x = ite_s.findSolution()
    #     #pretty_print_matrix_S(ite_x, "x.txt")
    #     with open("x_.npy", "wb") as f:
    #         np.save(f, ite_x)
    #     end = time.time()
    #     print("iterate method took:{0}s".format(end - start))
    #     sys.stdout.flush()

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
