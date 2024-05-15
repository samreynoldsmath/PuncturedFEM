"""
Preconditioners for iterative solvers.

Functions
---------
jacobi_preconditioner(A: LinearOperator) -> LinearOperator
    Get the Jacobi preconditioner for a linear operator A.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator


def jacobi_preconditioner(A: LinearOperator) -> LinearOperator:
    """
    Get the Jacobi preconditioner for a linear operator A.

    Parameters
    ----------
    A : LinearOperator
        Linear operator A

    Returns
    -------
    LinearOperator
        Jacobi preconditioner for A
    """
    # Jacobi preconditioner
    diagonals = np.zeros((A.shape[0],))
    ei = np.zeros((A.shape[0],))
    for i in range(A.shape[0]):
        ei[i] = 1
        diagonals[i] = np.dot(ei, A @ ei)
        ei[i] = 0

    # build preconditioner object
    return LinearOperator(
        dtype=float,
        shape=A.shape,
        matvec=lambda x: x / diagonals,
    )
