"""
legendre.py
===========

Module containing functions for working with Legendre Polynomials.

Routines in this module
-----------------------
legendre_polynomials(deg)
legendre_tensor_products(deg)
integrated_legendre_polynomials(deg)
integrated_legendre_tensor_products(deg)
swap_coordinates_of_poly_argument(p)
"""

from numpy import sqrt

from .poly import Polynomial


def legendre_polynomials(deg: int) -> list[Polynomial]:
    """
    Returns an array of Polynomial objects
            [p_0, p_1, ..., p_deg]
    where p_j is the jth Legendre Polynomial
    """
    polys = []
    polys.append(Polynomial([(1.0, 0, 0)]))
    polys.append(Polynomial([(1.0, 1, 0)]))
    for n in range(2, deg + 1):
        pn = (
            (2 * n - 1) * polys[1] * polys[n - 1] - (n - 1) * polys[n - 2]
        ) / n
        polys.append(pn)
    return polys


def legendre_tensor_products(deg: int) -> list[Polynomial]:
    """
    Returns the tensor products of the Legendre Polynomials up to degree deg as
    an array of Polynomial objects. The array is ordered in "triangular"
    fashion:
        0x0, 0x1, 0x2, ..., 0x(d-1), 0xd
        1x0, 1x1, 1x2, ..., 1x(d-1)
        ...
        (d-1)x0, (d-1)x1
        dx0
    which is read left to right, top to bottom.
    """
    p = legendre_polynomials(deg)
    q = []
    for pn in p:
        qn = swap_coordinates_of_poly_argument(pn)
        q.append(qn)
    pq = []
    for m in range(deg + 1):
        for n in range(deg + 1 - m):
            pq.append(p[m] * q[n])
    return pq


def integrated_legendre_polynomials(deg: int) -> list[Polynomial]:
    """
    Returns an array of Polynomial objects [q_0, q_1, ..., q_deg] where q_j is
    the jth integrated Legendre Polynomial
    """
    polys = []
    p = legendre_polynomials(deg)
    polys.append(p[0])
    polys.append(p[1])
    for n in range(2, deg + 1):
        qn = (p[n] - p[n - 2]) / sqrt(4 * n - 2)
        polys.append(qn)
    return polys


def integrated_legendre_tensor_products(deg: int) -> list[Polynomial]:
    """
    Returns the tensor products of the integrated Legendre Polynomials up to
    degree deg as an array of Polynomial objects. The array is ordered the same
    as in legendre_tensor_products().
    """
    p = integrated_legendre_polynomials(deg)
    q = []
    for pn in p:
        qn = swap_coordinates_of_poly_argument(pn)
        q.append(qn)
    pq = []
    for m in range(deg + 1):
        for n in range(deg + 1 - m):
            pq.append(p[m] * q[n])
    return pq


def swap_coordinates_of_poly_argument(p: Polynomial) -> Polynomial:
    """
    Returns the Polynomial q(x,y) = p(y,x)
    """
    q = Polynomial()
    for mono in p.monos:
        coef = mono.coef
        alpha_0 = mono.alpha.x
        alpha_1 = mono.alpha.y
        q += Polynomial([(coef, alpha_1, alpha_0)])
    return q
