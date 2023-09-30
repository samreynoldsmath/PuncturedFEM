"""
integrate_poly.py
=================

Module containing functions for integrating polynomials.

Routines in this module
-----------------------
integrate_poly_over_cell(p, K)

Notes
-----
The integral of a polynomial over a cell is computed by reducing this
volumetric integral to one on the boundary via the Divergence Theorem:
    int_K (x^alpha) dx = (1/|alpha|+2) int_{dK} (x^alpha)(x*n) dx
"""

from ...mesh.cell import cell
from .poly import polynomial


def integrate_poly_over_cell(p: polynomial, K: cell) -> float:
    """ "
    Returns the value of int_K (self) dx by reducing this volumetric integral
    to one on the boundary via the Divergence Theorem
    """
    x1, x2 = K.get_boundary_points()
    xn = K.dot_with_normal(x1, x2)
    val = 0.0
    for m in p.monos:
        integrand = xn * m.eval(x1, x2) / (2 + m.alpha.order)
        val += K.integrate_over_boundary(integrand)
    return val
