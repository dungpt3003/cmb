#!/usr/bin/env python
# Base class for solutions

import numpy as np
from utils import *
from matrix import *

class Solution:
    def __init__(self,afile,tfile,m,n,lvl,y):
        self.af = afile
        self.tf = tfile
        self.num_m = m
        self.num_n = n
        self.num_lvl = lvl
        self.num_N = calculate_N_from_level(lvl)

        # Init matrices
        self.A = None
        self.T = None
        self.load_matrices_from_files()
        
        self.B = generate_matrix_B(self.A, self.num_N)
        self.C = generate_matrix_C(self.T, self.num_N)
        self.D = generate_matrix_D(self.num_lvl)
        self.Q = generate_matrix_Q(self.D, self.num_n, self.num_N)
        self.u = None

        self.N = None
        self.C = None

        self.y = y
        self.u = None
        self.P = np.eye(self.num_n)
        self.Dsize = calc_D_size(self.num_N)
        
        
    def load_matrices_from_files(self):
        self.A = load_matrix(self.af)
        self.T = load_diagonal_matrix(self.tf)
    
    def get_D_size(self):
        return self.Dsize
        
    def get_D(self):
        return self.D

    def get_Q(self):
        return self.Q

    def get_B(self):
        return self.B

    def get_C(self):
        return self.C