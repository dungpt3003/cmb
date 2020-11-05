#!/usr/bin/env python

import time
import numpy as np
import sys
from utils import *
from conjugate_solution import ConjugateSolution

if __name__ == '__main__':
    m = 9
    n = 4
    sourcefilea = "data/a.csv"
    sourcefilet = "data/t.csv"
    tnum = 4

    for i in range(1,4):
        print("-------------lvl:{0}, N:{1}------------".format(i, calculate_N_from_level(i)))
        y = 5 * np.random.randint(0,10,(m * calculate_N_from_level(i), 1))
        #iterate method
        start = time.time()
        ite_s = ConjugateSolution(sourcefilea,sourcefilet,m,n,i,y,0.001,tnum)
        ite_x = ite_s.findSolution()
        end = time.time()
        print("iterate method took:{0}s".format(end - start))
        print("Solution:", ite_x)
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
