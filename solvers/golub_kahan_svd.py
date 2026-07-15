"""
Golub Reinsch Kahan: Calculating SVD through 'Givens' rotation of matrices.
This implementation follows the thesis from a swedish university's final year submission.
"""

import os
import math
import numpy as np
import numpy.typing as npt

EPS = 1e-12

def householder(X: npt.NDArray[np.float64]) -> np.ndarray:
    """Returns the householder reflection of a single vector"""
    X = X.copy()
    y = np.zeros_like(X)
    y[0] = -np.copysign(np.linalg.norm(X), X[0])
    v = X - y
    if np.linalg.norm(v) < EPS * 1e-2: return np.eye(len(X))
    tau = 2 / np.linalg.norm(v) ** 2   
    H   = np.eye(len(X)) - tau * np.outer(v, v)
    return H

def bidiag(A: npt.NDArray[np.float64]):
    """Bidiagonalizing a matrix for Golub Kahan steps"""
    X = A.copy()
    m, n = X.shape
    U = np.eye(m, dtype=float)
    V = np.eye(n, dtype=float)

    for k in range(n - 1):
        H         = householder(X[k:, k])
        X[k:, k:] = H @ X[k:, k:]
        U[:, k:]  = U[:, k:] @ H
        H         = householder(X[k, k + 1:])
        X[k:, k + 1:] = X[k:, k + 1:] @ H
        V[:, k + 1:]  = V[:, k + 1:] @ H

    # Zero columns that are less than EPS * 1e-3
    mask = abs(X) < EPS * 1e-3
    X[mask] = 0
    return X, U, V

def givens_coef(y: float, z: float) -> tuple[float, float]:
    """Calculates the Givens coefficient for two components of a vector"""
    if z == 0: return 1, 0
    if abs(z) >= abs(y):
        t = y / z
        s = 1 / ((1 + t**2) ** 0.5)
        c = s * t
    else:
        t = z / y
        c = 1 / ((1 + t**2) ** .5)
        s = c * t
    return c, s

def givens_prod(c, s, x, y):
    a = c * x + s * y
    b = -s * x + c * y
    return a, b

def wilkinson(B):
    n = B.shape[0]

    dm = B[n - 2, n - 2]
    fm = B[n - 2, n - 1]
    dn = B[n - 1, n - 1]

    T = np.array([
        [dm**2 + fm**2, dm * fm],
        [dm * fm, dn**2 + fm**2]
    ])

    eigvals = np.linalg.eigvalsh(T)
    target = dn**2 + fm**2
    mu = eigvals[np.argmin(np.abs(eigvals - target))]
    return mu

def gk_step(B: np.ndarray, U: np.ndarray, V: np.ndarray, offset: int):
    B = B.copy()
    n = B.shape[0]
    mu = wilkinson(B)
    
    y = B[0, 0]**2 - mu
    z = B[0, 0] * B[0, 1]

    for k in range(n - 1):
        c, s   = givens_coef(y, z)
        at     = B[:, k].copy()
        bt     = B[:, k + 1].copy()
        B[:, k], B[:, k + 1] = givens_prod(
            c, s,
            at,
            bt
        )
        V[:, offset + k], V[:, offset + k + 1] = givens_prod(
            c, s,
            V[:, offset + k].copy(),
            V[:, offset + k + 1].copy()
        )
        y    = B[k, k]
        z    = B[k + 1, k]
        c, s = givens_coef(y, z)
        at   = B[k, :].copy()
        bt   = B[k + 1, :].copy()
        B[k, :], B[k + 1, :] = givens_prod(
            c, s,
            at,
            bt
        )
        U[:, offset + k], U[:, offset + k + 1] = givens_prod(
            c, s,
            U[:, offset + k].copy(),
            U[:, offset + k + 1].copy()
        )
        if k < n - 2:
            y = B[k, k + 1]
            z = B[k, k + 2]

    return B, U, V

def active_block(B: np.ndarray):
    B = B.copy()
    n = B.shape[0]
    nz = [i for i in range(n - 1) if abs(B[i, i + 1]) > 0]
    if not nz:
        return None
    p = min(nz)
    q = p
    while q < n - 1 and abs(B[q, q + 1]) > 0: q += 1
    return p, q

def golub_kahan(A: np.ndarray):
    B, U, V = bidiag(A)
    n = B.shape[0]
    iteration = 0
    MAX_ITER = 1000
    while True:
        if iteration > MAX_ITER: break
        for i in range(n - 1):
            threshold = EPS * (abs(B[i, i]) + abs(B[i + 1, i + 1]))
            if abs(B[i, i + 1]) < threshold:
                B[i, i + 1] = 0.0
        block = active_block(B)
        if block is None:
            break
        p, q  = block
        A22   = B[p : q+1, p: q+1]        
        if np.any(np.abs(np.diag(A22)) < EPS): raise NotImplementedError("Zero diagonal error")
        A22, U, V           = gk_step(A22, U, V, offset=p)
        B[p : q+1, p : q+1] = A22
    
    return B, U, V
