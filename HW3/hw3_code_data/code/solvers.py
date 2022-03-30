"""
    Initially written by Ming Hsiao in MATLAB
    Rewritten in Python by Wei Dong (weidong@andrew.cmu.edu), 2021
"""

from scipy.sparse import csc_matrix, csr_matrix, eye
from scipy.sparse.linalg import inv, splu, spsolve, spsolve_triangular

from sparseqr import rz, permutation_vector_to_matrix, solve as qrsolve
import numpy as np
import matplotlib.pyplot as plt


def solve_default(A, b):
    from scipy.sparse.linalg import spsolve

    x = spsolve(A.T @ A, A.T @ b)
    return x, None


def solve_pinv(A, b):
    # TODO: return x s.t. Ax = b using pseudo inverse.
    N = A.shape[1]
    x = np.zeros((N,))

    x = inv(A.T @ A) @ A.T @ b
    return x, None


def solve_lu(A, b):
    # TODO: return x, U s.t. Ax = b, and A = LU with LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    # N = A.shape[1]
    # x = np.zeros((N,))
    # U = eye(N)
    lud = splu(csc_matrix(A.T @ A), permc_spec="NATURAL")
    x = lud.solve(A.T @ b)
    U = lud.U
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutration_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    # N = A.shape[1]
    # x = np.zeros((N,))
    # U = eye(N)
    lud = splu(csc_matrix(A.T @ A), permc_spec="COLAMD")
    x = lud.solve(A.T @ b)
    U = lud.U
    return x, U


def solve_qr(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |Rx - d|^2 + |e|^2
    # https://github.com/theNded/PySPQR
    # N = A.shape[1]
    # x = np.zeros((N,))
    # R = eye(N)
    z, R, E, rank = rz(A, b, permc_spec="NATURAL")
    x = spsolve_triangular(csr_matrix(R), z, lower=False)
    return x, R


def solve_qr_colamd(A, b):
    # TODO: return x, R s.t. Ax = b, and |Ax - b|^2 = |R E^T x - d|^2 + |e|^2, with reordered QR decomposition (E is the permutation matrix).
    # https://github.com/theNded/PySPQR
    # N = A.shape[1]
    # x = np.zeros((N,))
    # R = eye(N)
    z, R, E, rank = rz(A, b, permc_spec="COLAMD")
    E = permutation_vector_to_matrix(E)
    x = spsolve_triangular(R, z, lower=False)
    x = E @ x
    return x, R


# Bonus Implementation
def forward_substitution(L, b):
    y = np.zeros_like(b)
    num = y.shape[0]
    y[0] = b[0] / L[0, 0]
    for i in range(1, num):
        temp = b[i]
        for j in range(0, i):
            temp -= L[i, j] * y[j]
        y[i] = temp / L[i, i]
    return y


def backward_substitution(R, b):
    x = np.zeros_like(b)
    num = x.shape[0]
    x[num - 1] = b[num - 1] / R[num - 1, num - 1]
    for i in range(num - 2, -1, -1):
        temp = b[i]
        for j in range(i + 1, num):
            temp -= R[i, j] * x[j]
        x[i] = temp / R[i, i]
    return x


def solve_custom_lu_colamd(A, b):
    lu = splu(csc_matrix(A.T @ A), permc_spec="COLAMD")
    L = lu.L
    U = lu.U

    # Permutaion Matrices
    P = csc_matrix((np.ones(A.shape[1]), (lu.perm_r, np.arange(A.shape[1]))))
    Q = csc_matrix((np.ones(A.shape[1]), (np.arange(A.shape[1]), lu.perm_c)))

    # Custom forward and backward substitution (Slower than scipy's)
    z = forward_substitution(L, P @ A.T @ b)
    y = backward_substitution(U, z)
    x = Q @ y

    # In-built forward and backward substitution (Faster)
    # z = spsolve_triangular(L, P @ A.T @ b, lower=True)
    # y = spsolve_triangular(U, z, lower=False)
    # x = Q @ y

    return x, U


def solve(A, b, method="default"):
    """
    \param A (M, N) Jacobian matirx
    \param b (M, 1) residual vector
    \return x (N, 1) state vector obtained by solving Ax = b.
    """
    M, N = A.shape

    fn_map = {
        "default": solve_default,
        "pinv": solve_pinv,
        "lu": solve_lu,
        "qr": solve_qr,
        "lu_colamd": solve_lu_colamd,
        "qr_colamd": solve_qr_colamd,
        "custom_lu": solve_custom_lu_colamd,
    }

    return fn_map[method](A, b)
