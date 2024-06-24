import numpy as np
from matrix import *
from sylvester_efficient import *
from scipy.linalg import solve_sylvester
from utils import calculate_N_from_level
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import time
import json
from scipy.sparse import lil_matrix, identity
from scipy.sparse.linalg import eigsh
def get_shat(a, t):
    """
    Calculate SHat = A^T * T * A.
    """
    ret = a.T @ t @ a
    if ret.shape[0] != ret.shape[1]:
        raise ValueError("Matrix SHat should be symmetric, it should have equal rows and columns.")
    return ret

def valid_shat(U, egvals):
    """
    Check the validity of SHat by recomputing it using eigenvectors and eigenvalues.
    """
    R = np.diag(egvals)
    ret = U @ R @ U.T
    
def get_yhat(oriy, a, t, m, n, bign):
    """
    Compute YHat = Y * T * A.
    """
    Y = np.zeros((bign, n))
    for idx, val in enumerate(oriy):
        col = idx // bign
        row = idx % bign
        Y[row, col] = val
    
    yhat = Y @ t @ a
    return yhat

def symmetric_sparse_sylvester_solver(Y_tilde, N_star_1D_2, S_tilde, tol):
    # Placeholder for the solution M
    M = np.zeros_like(Y_tilde)
    
    # Initial calculations
    U, R = np.linalg.eig(S_tilde)  # Eigendecomposition
    Q, R = np.linalg.qr(Y_tilde)  # QR-factorization
    
    V_old = np.zeros_like(Q)
    i = 0
    residual_norm = np.inf
    
    # Lanczos iteration and residual norm computation
    while residual_norm > tol:
        i += 1
        
        # Lanczos step
        W = N_star_1D_2 @ Q
        if i > 1:
            W = W - V_old @ R[i]
        H_i = Q.T @ W
        W = W - Q @ H_i
        V_old = Q
        Q, R_new = np.linalg.qr(W)  # QR-factorization
        
        # Computing residual norm
        T_i = Q[i] @ R[i].T
        S = U.T @ R[0].T @ R[1] @ Q[i]
        J = Q[i] @ S @ R[i+1].T
        
        residual_norms = []
        for j in range(S_tilde.shape[1]):
            K = np.eye(S_tilde.shape[1]) + R[i]
            residual = np.linalg.norm(R[j].T @ S - K @ J)
            residual_norms.append(residual)
        residual_norm = np.mean(residual_norms)
    
    # Constructing approximation coefficients entry-wise
    F = np.zeros((Y_tilde.shape[1], Y_tilde.shape[1]))
    for k in range(Y_tilde.shape[1]):
        for l in range(Y_tilde.shape[1]):
            F[k, l] = R[k].T @ Q[l].T @ R[1] @ U @ R[l].T
    
    # Rerun Lanczos to assemble solution
    M = np.zeros_like(Y_tilde)
    V = Y_tilde @ R[0].T
    V_old = np.zeros_like(V)
    for j in range(i):
        M += V @ R[j]
        W = N_star_1D_2 @ V
        if j > 1:
            W = W - V_old @ R[j]
        W = W - V @ H_i
        V_old = V
        V = W @ R[j+1].T
    
    return M

def combined_sylvester_solver_with_N_star_1D_2(oriy, a, t, m, n, bign, tol, lvl):
    """
    A combined function to compute the solution to the Sylvester equations using the provided inputs.
    """
    # Compute YHat and SHat from the provided data
    YHat = get_yhat(oriy, a, t, m, n, bign)
    SHat = get_shat(a, t)
    
    # Compute N_star_1D_2
    N = identity(bign, format='csc')
    D = generate_matrix_D_efficient(lvl)  # This will need the actual parameters/arguments once we know its signature
    print("N shape: ", N.shape)
    print("D shape: ", D.shape)
    
    N_star_1D_2 = N @ (D.power(2))
    # N_star_1D_2 = sp.sparse.random(N.shape[0], N.shape[1], density=0.001)
    # plot_sparse_matrix(N_star_1D_2)
    print("Checkpoint 0, N-1D2 shape: ", N_star_1D_2.shape)
    # Solve the Sylvester equations to get the solution M
    start = time.time()
    #M_efficient = solve_sylvester_efficiently_corrected(N_star_1D_2, SHat, YHat)
    M_efficient = algorithm_3_sparse_dense_timed(N_star_1D_2, SHat, YHat)
    #M = symmetric_sparse_sylvester_solver(YHat, N_star_1D_2, SHat, tol)
    stop = time.time()
    #M_baseline = solve_sylvester(N_star_1D_2, SHat, YHat)
    #diff = np.linalg.norm(M_baseline - M_efficient, 'fro')
    print(f"Time for the case level = {lvl} is: {stop - start}")
    
    return M_efficient

A = np.loadtxt("data/a.csv", delimiter=",")
T = load_diagonal_matrix("data/t.csv")

def plot_sparse_matrix(S):
    plt.spy(S)
    plt.show()

def read_result_file(result_file):
    with open(result_file, "r") as f:
        data = f.read()
        result = [float(x) for x in data[1:-1].split(" ")]
    return np.array(result)

# Testing the combined function
for lvl in range(1,11):
   
    m, n, bign, lvl, tol = 4, 9, calculate_N_from_level(lvl) , lvl, 1e-9
    with open(f"data/y/vectory_{lvl}", "r") as file_y:
        y_content = file_y.read()
    y = np.array([float(val) for val in y_content.split(",")])
    M_solution = combined_sylvester_solver_with_N_star_1D_2(y, A, T, m, n, bign, tol, lvl)
    #M_baseline = read_result_file(f'output/output/result_syl_{lvl}')
    # diff = np.linalg.norm(M_solution.flatten() - base_solution, 'fro')
    # print(f'Residual for level = {lvl}: {diff}')
    # with open(f"output/efficient_syl_{lvl}.txt", 'w') as f:
    #     f.write(str(M_solution))