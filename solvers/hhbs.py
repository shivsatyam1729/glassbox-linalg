import math
import numpy as np
from linalg.norm import norm

def householder(X: np.array, eps: float=1e-12):
    """Householder QR"""
    # -> NOTE: was using n-1, changed to n for NxM where N != M
    X          = X.copy()
    m, n       = X.shape
    tau        = np.zeros(n, dtype=float)
    reflectors = []  # store (k, v) pairs so reconstruction can't get misaligned
    for k in range(n):
        xi    = X[k:, k]
        yi    = np.zeros_like(xi)
        yi[0] = -np.copysign(np.linalg.norm(xi), xi[0])
        v     = xi - yi
        if np.linalg.norm(v) < eps:
            continue  # degenerate column: leave tau[k]=0 (no-op reflector), skip cleanly
        tau[k]       = 2.0 / np.linalg.norm(v) ** 2
        # H = I - 2 \cdot vv^T
        v_col        = v.reshape(-1, 1)
        X[k:, k:]    = X[k:, k:] - tau[k] * v_col @ (v_col.T @ X[k:, k:])
        v_packed     = v / v[0]
        X[k + 1:, k] = v_packed[1:]
        reflectors.append((k, v))

    R = np.triu(X[:, :n])
    Q = np.eye(m)
    for k, v in reversed(reflectors):
        Hk          = np.eye(m)
        Hk[k:, k:] -= tau[k] * np.outer(v, v)
        Q           = Hk @ Q

    return Q, R

def lstsq(A: np.array, y: np.array) -> np.array:
    """Least squares solve of A @ beta ~= y via Householder-QR + back-substitution."""
    Q, R = householder(A)
    rhs  = Q.T @ y
    def bs(R: np.array, b: np.array) -> np.array:
        n = R.shape[0]
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            s    = b[i]
            for j in range(i + 1, n): s -= R[i, j] * x[j]
            x[i] = s / R[i, i]    
        return x
    beta = bs(R, rhs)
    return beta