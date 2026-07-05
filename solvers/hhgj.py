import numpy as np

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
    R_i = jordan(R, eps=1e-12)
    beta = R_i @ rhs
    return beta