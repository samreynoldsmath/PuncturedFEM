"""
barycentric.py
==============

Module containing functions for working with barycentric coordinates.

Routines in this module
-----------------------
barycentric_coordinates_edge(e)
barycentric_coordinates(z0, z1, z2)
barycentric_products(e, deg) # DEPRECATED
"""

from numpy import array, ndarray, sqrt, zeros
from numpy.linalg import norm

from ...mesh.edge import edge
from .poly import polynomial
from .poly_exceptions import DegenerateTriangleError

ROOT3OVER2 = sqrt(3) / 2


def barycentric_coordinates_edge(e: edge) -> list[polynomial]:
    """
    Returns the barcentric coordinates ell = [ell0, ell1, ell2]
    where ellj is a polynomial object and the point z2 is constructed
    from z0,z1 to form an equilateral triangle, with z0,z1 being the
    endpoints of the edge e.
    """
    z0 = e.x[:, 0]
    z1 = e.x[:, -1]
    R = array([[0, 1], [-1, 0]])
    z2 = 0.5 * (z1 + z0) - ROOT3OVER2 * R @ (z1 - z0)
    return barycentric_coordinates(z0, z1, z2)


def barycentric_coordinates(
    z0: ndarray, z1: ndarray, z2: ndarray
) -> list[polynomial]:
    """
    Returns the barycentric coordinates ell = [ell0, ell1, ell2]
    where ellj is a polynomial object
    """
    z = zeros((3, 2))
    z[0, :] = z0
    z[1, :] = z1
    z[2, :] = z2

    # compute distances
    dist = zeros((3,))
    for j in range(3):
        jminus1 = (j - 1) % 3
        dist[j] = norm(z[j] - z[jminus1])
        if dist[j] < 1e-12:
            raise DegenerateTriangleError("Degenerate triangle detected")

    # compute re-normalized altitudes
    s = sum(dist) / 2
    h = 2 * sqrt(s * (s - dist[0]) * (s - dist[1]) * (s - dist[2]))

    # initialize polynomial objects
    x = polynomial([(1.0, 1, 0)])
    y = polynomial([(1.0, 0, 1)])
    ell = [polynomial(), polynomial(), polynomial()]

    for j in range(3):
        jminus1 = (j - 1) % 3
        jplus1 = (j + 1) % 3
        ell[j] = (x - z[j, 0]) * (z[jminus1, 1] - z[jplus1, 1])
        ell[j] -= (y - z[j, 1]) * (z[jminus1, 0] - z[jplus1, 0])
        ell[j] *= -1 / h
        ell[j] += 1

    return ell


def barycentric_products(e: edge, deg: int) -> list[ndarray]:
    """
    DEPRECATED

    Returns a spanning set of the space of polynomials of degree deg <= 3 by
    computing the products of the barycentric coordinates
    """
    if deg > 3:
        raise NotImplementedError("Only implemented for deg <= 3")

    ell = barycentric_coordinates_edge(e)
    ell_trace = zeros((3, e.num_pts))
    for j in range(3):
        ell_trace[j, :] = ell[j].eval(x=e.x[0, :], y=e.x[1, :])

    spanning_trace = []
    for j in range(3):
        spanning_trace.append(ell_trace[j, :])

    # quadratics
    if deg > 1:
        for i in range(3):
            for j in range(i + 1, 3):
                spanning_trace.append(4 * ell_trace[i, :] * ell_trace[j, :])

    # cubics
    if deg > 2:
        spanning_trace.append(
            27 * ell_trace[0, :] * ell_trace[1, :] * ell_trace[2, :]
        )
        for i in range(3):
            for j in range(i + 1, 3):
                spanning_trace.append(
                    ROOT3OVER2
                    * ell_trace[i, :]
                    * ell_trace[j, :]
                    * (ell_trace[i, :] * ell_trace[j, :])
                )

    return spanning_trace
