"""
test_solver.py
==============

Test the solver class.
"""

import numpy as np
from scipy.sparse import csr_matrix

import puncturedfem as pf


TOL = 1e-6


def build_solver() -> pf.Solver:
    """
    Helper function: build a mesh and bilinear form, return a solver object.
    """

    # set test parameters
    n = 64
    interp = 4
    deg = 1
    a = 1.0
    c = 1.0
    f = pf.Polynomial([(1.0, 0, 0)])

    # define bilinear form
    B = pf.BilinearForm(
        diffusion_constant=a,
        reaction_constant=c,
        rhs_poly=f,
    )

    # build mesh
    T = pf.meshlib.pacman_subdiv(verbose=False)

    # build quadrature dictionary for parameterization
    quad_dict = pf.get_quad_dict(n=n, interp=interp)

    # build global function space
    V = pf.GlobalFunctionSpace(T, deg, quad_dict, verbose=False)

    # return solver object
    return pf.Solver(V, B)


def test_solver_deg1() -> None:
    """
    Test the solver class.
    """

    S = build_solver()
    S.assemble(verbose=False, compute_interior_values=False)
    S.solve()

    x_computed = S.soln
    x_exact = np.loadtxt("tests/data/glob_soln_n16.txt")
    x_error = x_exact - x_computed

    x_exact_h1_sq_norm = compute_h1_square_norm(
        x_exact, S.stiff_mat, S.mass_mat
    )
    x_error_h1_sq_norm = compute_h1_square_norm(
        x_error, S.stiff_mat, S.mass_mat
    )

    assert x_error_h1_sq_norm < TOL * x_exact_h1_sq_norm


def compute_h1_square_norm(
    x: np.ndarray, stiffness_mat: csr_matrix, mass_mat: csr_matrix
) -> float:
    """
    Returns x^t A x + x^t B x, where A,B are the stiffness and mass matrices,
    respectively.
    """
    h1 = np.dot(stiffness_mat @ x, x)
    l2 = np.dot(mass_mat @ x, x)
    return h1 + l2
