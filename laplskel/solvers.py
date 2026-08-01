"""Sparse linear-system solver implementations."""

import warnings

import numpy as np
from scipy.sparse.linalg import cg, spsolve

VALID_SOLVERS = ('LU', 'CG', 'AMGCG')


def validate_solver(solver):
    """Return a valid solver name or raise a user-facing error."""
    if solver not in VALID_SOLVERS:
        valid = ', '.join(VALID_SOLVERS)
        raise ValueError(f'Unknown solver {solver!r}. Expected one of: {valid}.')
    return solver


def _report_cg_status(info, solver, dimension):
    """Warn when a conjugate-gradient solve did not converge successfully."""
    if info > 0:
        warnings.warn(
            f'{solver} reached its iteration limit for coordinate dimension '
            f'{dimension} (info={info}).',
            RuntimeWarning,
            stacklevel=3,
        )
    elif info < 0:
        warnings.warn(
            f'{solver} failed for coordinate dimension {dimension} (info={info}).',
            RuntimeWarning,
            stacklevel=3,
        )


def _solve_cg(A, B, X, preconditioner=None, solver='CG'):
    """Solve all coordinate systems with an optional CG preconditioner."""
    X_next = np.zeros_like(X)
    for dimension in range(3):
        solution, info = cg(
            A,
            B[:, dimension],
            x0=X[:, dimension],
            M=preconditioner,
            rtol=1e-4,
            maxiter=500,
        )
        _report_cg_status(info, solver, dimension)
        X_next[:, dimension] = solution
    return X_next


def solve_lu(A, B):
    """Solve the three coordinate systems with sparse direct factorization."""
    X_next = np.zeros_like(B)
    for dim in range(3):
        X_next[:, dim] = spsolve(A, B[:, dim])
    return X_next


def solve_cg(A, B, X):
    """Solve the three coordinate systems with warm-started conjugate gradients."""
    return _solve_cg(A, B, X)


def _prepare_pyamg_matrix(A):
    """Return a CSR matrix whose index arrays are supported by PyAMG."""
    if A.indptr.dtype != np.int64 and A.indices.dtype != np.int64:
        return A

    maximum_index = max(A.shape[0], A.nnz)
    if maximum_index > np.iinfo(np.int32).max:
        return None

    compatible = A.copy()
    compatible.indptr = compatible.indptr.astype(np.int32)
    compatible.indices = compatible.indices.astype(np.int32)
    return compatible


def solve_amgcg(A, B, X):
    """Solve with AMG-preconditioned CG, falling back to CG when AMG is unavailable."""
    compatible = _prepare_pyamg_matrix(A)
    if compatible is None:
        warnings.warn(
            'AMGCG does not support this matrix index range; switching to CG.',
            RuntimeWarning,
            stacklevel=2,
        )
        return solve_cg(A, B, X), False

    try:
        import pyamg

        hierarchy = pyamg.ruge_stuben_solver(compatible)
        preconditioner = hierarchy.aspreconditioner(cycle='V')
    except ImportError:
        warnings.warn(
            'PyAMG is unavailable; switching to CG.',
            RuntimeWarning,
            stacklevel=2,
        )
        return solve_cg(A, B, X), False
    except (MemoryError, RuntimeError, TypeError, ValueError) as error:
        warnings.warn(
            f'AMGCG setup failed ({error}); switching to CG.',
            RuntimeWarning,
            stacklevel=2,
        )
        return solve_cg(A, B, X), False

    return _solve_cg(
        compatible,
        B,
        X,
        preconditioner=preconditioner,
        solver='AMGCG',
    ), True
