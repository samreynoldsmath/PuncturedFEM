"""
poly
====

This subpackage contains tools for working with polynomials in two variables.

Modules
-------
barycentric
    Functions for working with barycentric coordinates.
integrate_poly
    Functions for integrating polynomials.
legendre
    Functions for working with Legendre polynomials.
monomial
    Defines the monomial class.
multi_index
    Defines the multi_index class.
piecewise_polynomial
    Defines the piecewise_polynomial class.
poly_eval
    Functions for evaluating polynomials.
poly
    Defines the polynomial class.
"""

from .poly import polynomial

__all__ = ["polynomial"]
