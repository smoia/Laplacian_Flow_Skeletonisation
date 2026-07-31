"""Sparse linear-system solver implementations."""

import numpy as np
from scipy.sparse.linalg import cg, spsolve


def solve_lu(A, B):
    """Solve the three coordinate systems with sparse direct factorization."""
    X_next = np.zeros_like(B)
    for dim in range(3):
        X_next[:, dim] = spsolve(A, B[:, dim])
    return X_next


def solve_cg(A, B, X):
    """Solve the three coordinate systems with warm-started conjugate gradients."""
    X_next = np.zeros_like(X)
    for dim in range(3):
        # Use CG with the previous coordinate array as a warm start (x0)
        # rtol=1e-4 is plenty accurate for contraction steps
        sol, _ = cg(A, B[:, dim], x0=X[:, dim], rtol=1e-4, maxiter=500)
        X_next[:, dim] = sol
    return X_next
