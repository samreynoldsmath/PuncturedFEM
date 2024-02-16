"""
poly_eval.py
============

Module containing functions for evaluating Polynomials.

Routines in this module
-----------------------
get_poly_traces_edge(e, polys)
get_poly_vals(x, y, polys)
"""

import numpy as np

from ...mesh.edge import Edge
from ...mesh.mesh_exceptions import SizeMismatchError
from .poly import Polynomial


def get_poly_traces_edge(
    e: Edge, polys: list[Polynomial], interp: int
) -> np.ndarray:
    """
    Evaluates the Polynomials in polys on the Edge e.
    """
    ex, ey = e.get_sampled_points(interp)
    return get_poly_vals(x=ex, y=ey, polys=polys)


def get_poly_vals(
    x: np.ndarray, y: np.ndarray, polys: list[Polynomial]
) -> np.ndarray:
    """
    Evaluates the Polynomials in polys at the points (x[j], y[j]).
    """
    num_pts = len(x)
    if len(y) != num_pts:
        raise SizeMismatchError("x and y must be the same length")
    num_polys = len(polys)
    poly_vals = []
    for j in range(num_polys):
        poly_vals.append(polys[j].eval(x, y))
    return np.array(poly_vals)
