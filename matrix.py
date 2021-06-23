#!/usr/bin/env python
# This files contains functions to construct matrices 
import numpy as np
from utils import calculate_matrix_A, find_neighbors, calc_D_size, calculate_N_from_level

# Load normal matrix from file
def load_matrix(aFile):
    with open(aFile, "r") as f:
        A = np.loadtxt(f, delimiter=",")
        return A

# Load values from file and create a diagonal matrix
def load_diagonal_matrix(aFile):
    with open(aFile, "r") as f:
        temp = np.loadtxt(f,delimiter=",")
        T = np.zeros((temp.shape[0], temp.shape[0]))
        for i in range(0,temp.shape[0]):
            T[i,i] = temp[i]
        return T

# Load values from file and create a diagonal matrix
def load_diagonal_matrix_inverse(aFile):
    with open(aFile, "r") as f:
        temp = np.loadtxt(f,delimiter=",")
        T = np.zeros((temp.shape[0], temp.shape[0]))
        for i in range(0,temp.shape[0]):
            T[i,i] = 1/temp[i]
        return T

# this is used in evaluation.py to load t.csv and then this is used to generate the diagonals
# of the precision matrix C
def load_array(aFile):
    with open(aFile, "r") as f:
       return np.loadtxt(f,delimiter=",")

# Compute the mat-vec product Bs utilising the Kronecker structure of B
def Bs(s, A, N):
    S = np.reshape(s, (N, A.shape[1]), order='F')
    BS = np.dot(S, np.transpose(A))
    y = np.reshape(BS, (-1,), order='F')
    return y
    
# Generate matrix matrix B as the Kronecker product of A and the identity matrix size N
def generate_matrix_B(A, N):
    return np.kron(A, np.eye(N))

# Generate matrix C as the Kronecker product of two diagonal matrices (T and diagN)
def generate_matrix_C(T, N):
    diagN = np.eye(N)
    return np.kron(T, diagN)

# Generate matrix D
def generate_matrix_D(lvl):
    size = calculate_N_from_level(lvl)
    (m,n) = calc_D_size(lvl)
    res = np.zeros((size,size),dtype=int)

    for i in range(0, size):
        neighbors = find_neighbors(i, m, n) 
        for pos in neighbors:
            res[i][pos] = 1
        
        res[i][i] = -1 * len(neighbors)
    return res

# Generate matrix Q as Kronecker product of diagonal matrix P and D square
def generate_matrix_Q(D, n, N):     
    qsize = n * N
    Q = np.zeros((qsize,qsize), dtype = int)
    DSqure = D.dot(D)
    for s in range(0, qsize, N):
        Q[s : s + N, s : s + N] = DSqure
    Q = Q
    return Q

# Generate matrix S_hat = A^TTAP^{-1}, assumming that P is a identity matrix
def generate_matrix_S_hat(A, T):
    A_trans = np.transpose(A)
    return A_trans.dot(T.dot(A))

# Generate matrix Y_hat = YTAP^{-1}, with Y = reshape(y,N,m)
def generate_matrix_Y_hat(y, N, m, T, A):
    Y = np.reshape(y,(N, m), order='F')

    return Y.dot(T.dot(A))
