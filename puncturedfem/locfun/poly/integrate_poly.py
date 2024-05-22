"""
Integrating Polynomials.

Routines in this module
-----------------------
integrate_poly_over_mesh(p, K)
"""

from ...mesh.cell import MeshCell
from .poly import Polynomial


def integrate_poly_over_mesh_cell(p: Polynomial, K: MeshCell) -> float:
    """
    Get the integral of a Polynomial over a MeshCell.

    Parameters
    ----------
    p : Polynomial
        Polynomial to integrate.
    K : MeshCell
        MeshCell over which to integrate.

    Returns
    -------
    float
        Integral of the Polynomial over the MeshCell.

    Notes
    -----
    The integral of a Polynomial over a MeshCell is computed by reducing this
    volumetric integral to one on the boundary via the Divergence Theorem:
        int_K (x^alpha) dx = (1/|alpha|+2) int_{dK} (x^alpha)(x*n) dx
    """
    x1, x2 = K.get_boundary_points()
    xn = K.dot_with_normal(x1, x2)
    val = 0.0
    for m in p.monos:
        integrand = xn * m.eval(x1, x2) / (2 + m.alpha.order)
        val += K.integrate_over_boundary(integrand)
    return val
