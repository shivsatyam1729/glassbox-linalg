import numpy as np
from linalg.norm import norm

def mgs_qr(X: np.array):
    """Modified Gram-Schmidt QR decomposition. X = Q @ R, Q orthonormal columns,
    R upper triangular. Assumes X has full column rank."""
    Q, R     = np.zeros_like(X, dtype=float), np.zeros((X.shape[1], X.shape[1]))
    Q[:, 0]  = X[:, 0]
    R[0, 0]  = np.linalg.norm(Q[:, 0])
    Q[:, 0] /= R[0, 0]
    for i in range(1, X.shape[1]):
        Q[:, i] = X[:, i]
        for k in range(i):
            R[i, k]  = np.dot(X[:, i], Q[:, k])
            u        = R[i, k] * Q[:, k]
            Q[:, i] -= u
        R[i, i]  = np.linalg.norm(Q[:, i])
        Q[:, i] /= R[i, i]
 
    return Q, R.T

def lstsq(A: np.array, y: np.array) -> np.array:
    """Least squares solve of A @ beta ~= y via MGS-QR + back-substitution."""
    Q, R = mgs_qr(A)
    rhs  = Q.T @ y
    # For positive semi definite UT matrix, use bs
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