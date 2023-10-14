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
    Build a mesh and bilinear form.
    """

    n = 16
    a = 1.0
    c = 1.0
    f = pf.Polynomial([[1.0, 0, 0]])

    T = pf.meshlib.pacman_subdiv(verbose=False)

    q_trap = pf.Quad(qtype="trap", n=n)
    q_kress = pf.Quad(qtype="kress", n=n)
    quad_dict = {"kress": q_kress, "trap": q_trap}

    V = pf.GlobalFunctionSpace(T, deg=1, quad_dict=quad_dict, verbose=False)

    B = pf.BilinearForm(
        diffusion_constant=a,
        reaction_constant=c,
        rhs_poly=f,
    )

    return pf.Solver(V, B)


def test_solver_deg1():
    """
    Test the solver class.
    """

    # TODO: test assemble and solve methods separately

    S = build_solver()
    S.assemble(verbose=False, compute_interior_values=False)
    S.solve()

    x = np.loadtxt("tests/data/glob_soln_n16.txt")
    assert np.linalg.norm(S.soln - x) < TOL
