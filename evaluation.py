#!/usr/bin/env python

import time
import numpy as np
import sys
from matrix import *
from utils import *
from conjugate_solution import ConjugateSolution
from standard_solution import StandardSolution
import sys, getopt
import tracemalloc

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
    #B = generate_matrix_B(A, N)
    #T = load_diagonal_matrix("data/t.csv")
    T1 = load_array("data/t.csv")
    #C = generate_matrix_C(T, N)
    s = np.random.normal(mean_s, sd_s, m*N)

    #mean_y = np.matmul(B, s)
    mean_y1 = Bs(s, A, N)
    #sd = np.array([1/np.sqrt(x) for x in np.diag(C)])
    sd1 = np.array([1/np.sqrt(T1[i // N]) for i in range(N * n)])
    #y = np.random.normal(mean_y, sd, mean_y.shape[0])
    y = np.random.normal(mean_y1, sd1, mean_y1.shape[0])
    if printable:
        np.savetxt('output/random_y_level_{0}.txt'.format(h), [y], delimiter=',', fmt='%0.20f')
        np.savetxt('output/random_s_level_{0}.txt'.format(h), [s], delimiter=',', fmt='%0.20f')

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
    method = 'cg'
    data = 'planck'
    try:
        opts, args = getopt.getopt(argv,"hl:m:d:",["level=","method=", "data="])
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
        elif opt in ("-d", "--data"):
            data = arg
    for l in range(level, 11):
        tracemalloc.start()
        if data == 'planck':
            if l < 10:
                y = np.loadtxt("data/y/vectory_{0}".format(l), delimiter=",")
            else:
                y = np.array([])
                for i in range(4):
                    temp = np.loadtxt("vectory_10_{0}".format(i), delimiter=",")
                    y = np.concatenate((y, temp))
            print(y.shape)
        else:
            y, _ = generate_random_sources(n, m, l, 1, 0.1, True)
        start = time.time()
        if method == 'cg':
            method_s = ConjugateSolution(source_file_a, source_file_t,  n,  m, l, y, 0.001, thread_num)
            method_x = method_s.findSolution()
            np.savetxt('output/result_s_cg_level_{0}.txt'.format(l), [method_x], delimiter=',', fmt='%0.20f')
        else:
            method_s = StandardSolution(source_file_a, source_file_t, n, m, l, y)
            method_x = method_s.findSolution()
            np.savetxt('output/result_s_std_level_{0}.txt'.format(l), [method_x], delimiter=',', fmt='%0.20f')
        end = time.time()
        with open("output/log.txt", "a") as f:
            f.write("The {0} method took {1} seconds in level {2} on {3} data\n".format(method, end - start, l, data))
        mems = tracemalloc.get_traced_memory()
        with open("output/log.txt", "a") as f:
            f.write("Current memory usage: {0}, Peak Memory Usage: {1}\n".format(mems[0] / float(1000000), mems[1]/float(1000000)))
        tracemalloc.stop()
    
    
if __name__ == '__main__':
    main(sys.argv[1:])