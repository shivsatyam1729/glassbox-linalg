import math
import numpy as np

def norm(X: np.array) -> float:
    if isinstance(X, np.ndarray): X = X.tolist()
    if len(X) > 0 and not isinstance(X[0], list): X = [X]
    
    if not (len(X) == 1 or len(X[0]) == 1):
        raise ValueError("Only Nx1, 1xN supported...")

    # sum(v*v) is implemented with C under the hood
    flat = X[0] if len(X) == 1 else [row[0] for row in X]
    return math.sqrt(sum(v * v for v in flat)) # <- math.sqrt is a C level call
