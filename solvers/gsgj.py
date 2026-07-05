import math
import numpy as np

def mgs_qr(X: np.array):
    """Modified Gram-Schmidt QR decomposition. X = Q @ R, Q orthonormal columns,
    R upper triangular. Assumes X has full column rank."""
    Q, R = np.zeros_like(X, dtype=float), np.zeros((X.shape[1], X.shape[1]))
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
    """Least squares solve of A @ beta ~= y via MGS-QR + Gauss Jordan Elimination."""
    Q, R = mgs_qr(A)
    rhs  = Q.T @ y

    def jordan(A: np.array, eps: float) -> np.array:
        n = len(A)
        A = list(A)
        # -> FIX: from row + [1.0....] to list(row) + [1.0...]
        aug = [list(row) + [1.0 if i == j else 0.0 for j in range(n)]
           for i, row in enumerate(A)]
    
        for i in range(n):
            # --- Partial pivoting: find row with largest value in this column ---
            pivot_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
            if aug[pivot_row][i] < eps:
                raise ValueError("Matrix is singular!")

            aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

            # --- Normalize pivot row so pivot becomes 1 ---
            pivot_val = aug[i][i]
            aug[i]    = [x / pivot_val for x in aug[i]]

            # --- Eliminate this column in all other rows ---
            for r in range(n):
                if r != i:
                    factor = aug[r][i]
                    aug[r] = [aug[r][k] - factor * aug[i][k] for k in range(2 * n)]
        
        gauss = [row[n:] for row in aug]
        return gauss
    
    R_i  = jordan(R, eps=1e-14)
    beta = R_i @ rhs
    return beta