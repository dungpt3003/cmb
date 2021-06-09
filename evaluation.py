#!/usr/bin/env python

import time
import numpy as np
import sys
from matrix import *
from utils import *
from conjugate_solution import ConjugateSolution
from standard_solution import StandardSolution
import sys, getopt

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

    return y, s

def main(argv):
    # The number of frequencies and number of sources for CMB problem are fixed
    n = 9
    m = 4
    source_file_a = "data/a.csv"
    source_file_t = "data/t.csv"
    thread_num = 4

    # Define command line arguments: HEALPix level and method used.
    level = 1
    method = 'std'
    try:
        opts, args = getopt.getopt(argv,"hl:m:",["level=","method="])
    except getopt.GetoptError:
        print ('evaluation.py -l <level> -m <method>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('evaluation.py -l <level> -m <method>')
            sys.exit()
        elif opt in ("-l", "--level"):
            level = int(arg)
        elif opt in ("-m", "--method"):
            method = arg
    y, s = generate_random_sources(n, m, level, 1, 0.1, True)
    start = time.time()
    if method == 'cg':
        method_s = ConjugateSolution(source_file_a, source_file_t,  n,  m, level, y, 0.001, thread_num)
        method_x = method_s.findSolution()
        np.savetxt('output/result_s_cg.txt', [method_x], delimiter=',', fmt='%0.20f')
    else:
        method_s = StandardSolution(source_file_a, source_file_t, n, m, level, y)
        method_x = method_s.findSolution()
        np.savetxt('output/result_s_std.txt', [method_x], delimiter=',', fmt='%0.20f')
    end = time.time()
    print("The {0} method took {1} seconds".format(method, end - start))
    distance = s - method_x
    print("Distance square:", np.dot(distance,distance))
    
if __name__ == '__main__':
    main(sys.argv[1:])
