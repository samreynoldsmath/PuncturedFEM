"""
test_solver.py
==============

Test the solver class.
"""

import numpy as np

import puncturedfem as pf

TOL = 1e-12


def build_solver() -> pf.Solver:
    """
    Helper function: build a mesh and bilinear form, return a solver object.
    """

    # set test parameters
    n = 16
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
    quad_dict = pf.get_quad_dict(n)

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

    h1_error = np.dot(S.stiff_mat @ x_error, x_error)
    l2_error = np.dot(S.mass_mat @ x_error, x_error)

    assert h1_error + l2_error < TOL
