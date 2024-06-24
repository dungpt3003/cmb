import numpy as np
from scipy.linalg import solve_sylvester
from scipy.linalg import schur, qr
from scipy.sparse.linalg import spsolve, SuperLU
from scipy.sparse import csc_matrix, eye, identity
from scipy.sparse.linalg import cg, gmres, splu
from os import environ
import matplotlib.pyplot as plt 
environ['OMP_NUM_THREADS'] = '4'
import time


def solve_sylvester_equation(H, S, Q):
    """
    Solve the Sylvester equation HM + MS = Q
    
    Parameters:
    H (2D array): Large, sparse matrix
    S (2D array): Small, dense matrix
    Q (2D array): Right-hand side matrix
    
    Returns:
    M (2D array): Solution matrix
    """
    M = solve_sylvester(H, S, Q)
    return M

def solve_sylvester_efficiently_corrected(H, S, Q):
    """
    Solve the Sylvester equation HM + MS = Q using the method described in point 5.
    
    Parameters:
    H (csc_matrix): Large, sparse matrix
    S (ndarray): Small, dense matrix
    Q (ndarray): Right-hand side matrix
    
    Returns:
    M (ndarray): Solution matrix
    """
    
    # Step 1: Compute the Schur form of S
    
    R, W = schur(S)

    Q_ = Q @ W
    print("Checkpoint 2")
    # Dimensions
    m = S.shape[0]
    n = H.shape[0]
    I_m = np.eye(m)
    print(S.shape, H.shape)
    # Initialize Y matrix
    MW = np.zeros((n, m))
    print("Checkpoint 3")
    for i in range(m):
        # Step 2: Solve the Sylvester equation
        
        H_ = H + R[i, i] * np.eye(n)
        print("Checkpoint 4")
        
        right = Q_[:, i]

        print("Checkpoint 5")
        for j in range(i):
            right -= MW[:, j] * R[j, i]
        print("Checkpoint 6")
        #MW[:, i] = spsolve(H_, right)
        #MW[:, i] = np.linalg.solve(H_, right)
        #MW[:, i], _ = cg(H_, right)

    # Step 3: Compute M
    M = MW @ W.T
    print("Checkpoint 7")
    
    return M

def sparse_dense_sylvester_solver(A, H, M):
    """
    Solve AX + XH = -M where A is a sparse matrix, H is a dense matrix, and M is a dense matrix.

    Parameters:
    A (scipy.sparse.csc_matrix): Sparse matrix A of shape (n, n)
    H (numpy.ndarray): Dense matrix H of shape (r, r)
    M (numpy.ndarray): Dense matrix M of shape (n, r)

    Returns:
    X (numpy.ndarray): Solution to the equation of shape (n, r)
    """
    # if not isinstance(A, csc_matrix):
    #     raise ValueError("A must be a scipy.sparse.csc_matrix")
    
    n, r = M.shape
    S, U = schur(H)
    print(f"Checkpoint 1: S shape: {S.shape}")
    
    # Step 1: Transform M
    M_tilde = M @ U
    print("Checkpoint 2")

    # Initialize X_tilde
    X_tilde = np.zeros((n, r))
    
    # Solve for X_tilde
    for j in range(r):
        rhs = -M_tilde[:, j]
        for i in range(j):
            rhs -= S[i, j] * X_tilde[:, i]
        X_tilde[:, j] = splu(A + S[j, j] * csc_matrix(np.eye(n))).solve(rhs)
        B = splu(A)
        L, R = B.L, B.U
        plot_sparse_matrix(L)
        plot_sparse_matrix(R)
        print(f"Checkpoint 3.{j}")
    # Transform X_tilde back to X
    X = X_tilde @ U.T
    print("Checkpoint 4")
    return X

def algorithm_3_sparse_dense_optimized(A, H, M):
    """
    Solve AX + XH = -M where A is a sparse matrix, H is a dense matrix, and M is a dense matrix.

    Parameters:
    A (scipy.sparse.csc_matrix): Sparse matrix A of shape (n, n)
    H (numpy.ndarray): Dense matrix H of shape (r, r)
    M (numpy.ndarray): Dense matrix M of shape (n, r)

    Returns:
    X (numpy.ndarray): Solution to the equation of shape (n, r)
    """
    # if not isinstance(A, csc_matrix):
    #     raise ValueError("A must be a scipy.sparse.csc_matrix")
    
    n, r = M.shape
    U, S = schur(H, output='complex')
    
    # Step 1: Transform M
    M_tilde = M @ U

    # Precompute LU decompositions
    lu_factors = []
    I = identity(n, format='csc')
    for j in range(r):
        lu_factors.append(splu(A + S[j, j] * I))
    
    # Initialize X_tilde
    X_tilde = np.zeros((n, r), dtype=complex)
    
    # Solve for X_tilde
    for j in range(r):
        rhs = -M_tilde[:, j]
        for i in range(j):
            rhs -= S[i, j] * X_tilde[:, i]
        X_tilde[:, j] = lu_factors[j].solve(rhs)
    
    # Transform X_tilde back to X
    X = X_tilde @ U.conj().T
    
    return np.real(X)

def calculate_rank_qr(M, tol=1e-10):
    """
    Calculate the numerical rank of a matrix M using QR decomposition with column pivoting.
    
    Parameters:
    M (numpy.ndarray): Matrix to calculate the rank of
    tol (float): Tolerance for singular values to be considered non-zero
    
    Returns:
    int: Numerical rank of the matrix M
    """
    Q, R, P = qr(M, pivoting=True)
    rank = np.sum(np.abs(np.diag(R)) > tol)
    return rank


def algorithm_3_sparse_dense_timed(A, H, M):
    """
    Solve AX + XH = -M where A is a sparse matrix, H is a dense matrix, and M is a dense matrix.
    This function includes timers to measure the complexity of each step in Algorithm 3.

    Parameters:
    A (scipy.sparse.csc_matrix): Sparse matrix A of shape (n, n)
    H (numpy.ndarray): Dense matrix H of shape (r, r)
    M (numpy.ndarray): Dense matrix M of shape (n, r)

    Returns:
    X (numpy.ndarray): Solution to the equation of shape (n, r)
    timings (dict): Dictionary with timings for each step
    """
    if not isinstance(A, csc_matrix):
        raise ValueError("A must be a scipy.sparse.csc_matrix")
    
    rank = calculate_rank_qr(M)
    print("Rank: ", rank)
    n, r = M.shape
    timings = {}

    # Step 1: Schur Decomposition
    start_time = time.time()
    U, S = schur(H, output='complex')
    timings['Step 1: Schur Decomposition'] = time.time() - start_time
    
    # Step 2: Transform M
    start_time = time.time()
    M_tilde = M @ U
    timings['Step 2: Transform M'] = time.time() - start_time

    # Step 4: Precompute LU decompositions
    start_time = time.time()
    lu_factors = []
    I = identity(n, format='csc')
    for j in range(r):
        lu_factors.append(splu(A + S[j, j] * I))
    timings['Step 3: Precompute LU'] = time.time() - start_time
    
    # Step 5: Solve for X_tilde
    X_tilde = np.zeros((n, r), dtype=complex)
    start_time = time.time()
    for j in range(r):
        rhs = -M_tilde[:, j]
        for i in range(j):
            rhs -= S[i, j] * X_tilde[:, i]
        X_tilde[:, j] = lu_factors[j].solve(rhs)
    timings['Step 5: Solve for X~'] = time.time() - start_time
    
    # Step 7: Transform X_tilde back to X
    start_time = time.time()
    X = X_tilde @ U.conj().T
    timings['Step 7: Transform X_tilde back to X'] = time.time() - start_time
    
    with open("timings.txt", "a") as f:
        f.write(str(timings))
        f.write("\n")
    return np.real(X)

def test_methods(n, m):
    # Generate random matrices
    H = csc_matrix(np.diag(np.random.rand(n)))
    S = np.random.rand(m, m)
    Q = np.random.rand(n, m)
    
    # Solve using baseline method
    M_baseline = solve_sylvester(H.toarray(), S, Q)
    
    # Solve using efficient method
    M_efficient = solve_sylvester_efficiently_corrected(H, S, Q)
    #M_baseline = M_efficient
    # Compute the Frobenius norm of the difference
    diff = np.linalg.norm(M_baseline - M_efficient, 'fro')
    
    return diff

def plot_sparse_matrix(S):
    plt.spy(S)
    plt.show()

if __name__ == '__main__':
    # Test for different sizes
    sizes = [(100, 2), (200, 3), (300, 4), (400, 5)]
    diffs = [test_methods(n, m) for n, m in sizes]
    print(sizes, diffs)

    