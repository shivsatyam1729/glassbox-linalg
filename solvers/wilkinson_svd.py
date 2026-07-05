import numpy as np
from linalg.norm import norm
from sklearn.datasets import make_regression

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

def qr_eig(M: np.array, iters=500, tol=1e-12):
    """QR for eigenvalues with Wilkinson's shift + deflation"""
    A        = M.copy()
    n        = A.shape[0]
    V        = np.eye(n)
    m_active = n
    it_used  = 0
    
    for i in range(iters):
        it_used += 1
        if m_active <= 1: break

        a, b = A[m_active - 2, m_active - 2], A[m_active - 1, m_active - 1]
        bc   = A[m_active - 2, m_active - 1]

        # deflate: last row of the active block has already converged
        last_row_off = A[m_active - 1, :m_active - 1]
        if np.max(np.abs(last_row_off)) < tol * (abs(b) + 1.0):
            m_active -= 1
            continue

        d     = (a - b) / 2.0
        s     = 1.0 if d >= 0 else -1.0
        denom = d + s * np.sqrt(d ** 2 + bc ** 2)
        mu    = b - (bc ** 2 / denom)
        
        Asub                    = A[:m_active, :m_active]
        Q, R                    = householder(Asub - mu * np.eye(m_active))
        A[:m_active, :m_active] = R @ Q + mu * np.eye(m_active)  # add mu back!
        V[:, :m_active]         = V[:, :m_active] @ Q
    
    return np.diag(A), V

def svd(A: np.array):
    """Inefficient SVD without bidigaonalization""" 
    X      = A.copy() 
    m      = X.T @ X
    eig, V = qr_eig(m)
    idx    = np.argsort(eig)[::-1]
    eig    = eig[idx]
    V      = V[:, idx]
    sigma  = np.sqrt(np.maximum(eig, 0))
    tol    = 1e-5
    r      = np.sum(sigma > tol)
    sigma  = sigma[:r]
    V      = V[:, :r]
    U      = np.zeros((A.shape[0], r))
    for i in range(r):
        u       = (A @ V[:, i]) / sigma[i]
        U[:, i] = u / np.linalg.norm(u)
    
    return U, sigma, V.T
   
def lstsq(A: np.array, y: np.array) -> np.array:
    """Least squares solve of A @ beta ~= y via SVD + QR + numpy's inv."""
    U, E, Vt = svd(A)
    E_i      = np.linalg.pinv(np.diag(E))
    beta     = Vt.T @ E_i @ U.T @ y
    return beta

if __name__ == "__main__":
    np.random.seed(42)

    print("--- SVD TEST ---")
    X = np.random.randn(100, 10)
    U, s, Vt = svd(X)
    recon_err = np.linalg.norm(X - U @ np.diag(s) @ Vt)
    if recon_err < 1e-8: print(f"PASSED | reconstruction error = {recon_err:.3e}")
    else:
        print(f"FAILED | reconstruction error = {recon_err:.3e}")

    print()
    print("--- LEAST SQUARES TEST ---")
    X, y = make_regression(
        n_samples    = 10000,
        n_features   = 10,
        noise        = 10.0,
        random_state = 42
    )
    beta_ours   = lstsq(X, y)
    beta_np, *_ = np.linalg.lstsq(X, y, rcond=None)

    beta_err      = np.linalg.norm(beta_ours - beta_np)
    pred_err      = np.linalg.norm(X @ beta_ours - X @ beta_np)
    residual_ours = np.linalg.norm(X @ beta_ours - y)
    residual_np   = np.linalg.norm(X @ beta_np - y)

    if pred_err < 1e-6: print(f"PASSED | prediction error = {pred_err:.3e}")
    else:
        print(f"FAILED | prediction error = {pred_err:.3e}")

    print()
    print(f"Beta error      : {beta_err:.3e}")
    print(f"Prediction error: {pred_err:.3e}")
    print(f"Residual ours   : {residual_ours:.3e}")
    print(f"Residual numpy  : {residual_np:.3e}")