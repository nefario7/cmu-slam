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
    A = csc_matrix(A)
    lud = splu(A.T @ A, permc_spec="NATURAL")
    x = lud.solve(A.T @ b)
    U = lud.U.A
    return x, U


def solve_lu_colamd(A, b):
    # TODO: return x, U s.t. Ax = b, and Permutation_rows A Permutration_cols = LU with reordered LU decomposition.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html
    # N = A.shape[1]
    # x = np.zeros((N,))
    # U = eye(N)
    A = A.toarray()
    A = csc_matrix(A)
    lud = splu(A.T @ A, permc_spec="COLAMD")
    x = lud.solve(A.T @ b)
    U = lud.U.A
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
def lu_decomp(A):
    """(L, U) = lu_decomp(A) is the LU decomposition A = L U
    A is any matrix
    L will be a lower-triangular matrix with 1 on the diagonal, the same shape as A
    U will be an upper-triangular matrix, the same shape as A
    """
    n = A.shape[0]
    if n == 1:
        L = np.array([[1]])
        U = A.copy()
        return (L, U)

    A11 = A[0, 0]
    A12 = A[0, 1:]
    A21 = A[1:, 0]
    A22 = A[1:, 1:]

    L11 = 1
    U11 = A11

    L12 = np.zeros(n - 1)
    U12 = A12.copy()

    L21 = A21.copy() / U11
    U21 = np.zeros(n - 1)

    S22 = A22 - np.outer(L21, U12)
    (L22, U22) = lu_decomp(S22)

    L = np.block([[L11, L12], [L21, L22]])
    U = np.block([[U11, U12], [U21, U22]])
    return (L, U)


def solve_custom_lu(A, b):
    A = A.toarray()

    print("LU decomposition")
    (L, U) = lu_decomp(A)
    print("Forward and backward substitution")
    x = np.zeros_like(b)
    # # forward
    # y = np.zeros_like(b)
    # for i in range(L.shape[0]):
    #     t = b[i]
    #     for j in range(i - 1):
    #         t -= L[i, j] * x[j]
    #     x[i] = t / L[i, i]

    # # backward
    # x = np.zeros_like(b)
    # for i in range(L.shape[0] - 1, -1, -1):
    #     tmp = y[i]
    #     for j in range(i + 1, n):
    #         tmp -= U[i, j] * x[j]
    #     x[i] = tmp / U[i, i]

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
        "custom_lu": solve_custom_lu,
    }

    return fn_map[method](A, b)
