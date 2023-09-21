import numpy as np
from matrix import generate_matrix_D, load_diagonal_matrix

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
    print("Recomputed S Hat:")
    print(ret)

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
    N = np.eye(bign)
    D = generate_matrix_D(lvl)  # This will need the actual parameters/arguments once we know its signature
    N_star_1D_2 = np.linalg.inv(N) @ np.square(D)
    
    # Solve the Sylvester equations to get the solution M
    M = symmetric_sparse_sylvester_solver(YHat, N_star_1D_2, SHat, tol)
    
    return M

A = np.loadtxt("data/a.csv", delimiter=",")
T = load_diagonal_matrix("data/t.csv")
with open("data/y/vectory_1", "r") as file_y:
    y_content = file_y.read()
y = np.array([float(val) for val in y_content.split(",")])

# Testing the combined function
m, n, bign, lvl, tol = 4, 9, 48, 1, 1e-9
M_solution = combined_sylvester_solver_with_N_star_1D_2(y, A, T, m, n, bign, tol, lvl)
print(M_solution)