"""
poly
====

This subpackage contains tools for working with Polynomials in two variables.

Modules
-------
barycentric
    Functions for working with barycentric coordinates.
integrate_poly
    Functions for integrating Polynomials.
legendre
    Functions for working with Legendre Polynomials.
monomial
    Defines the Monomial class.
multi_index
    Defines the multi_index class.
PiecewisePolynomial
    Defines the PiecewisePolynomial class.
poly_eval
    Functions for evaluating Polynomials.
poly
    Defines the Polynomial class.
"""

from .piecewise_poly import PiecewisePolynomial
from .poly import Polynomial

__all__ = ["Polynomial", "PiecewisePolynomial"]
