"""
poly_eval.py
============

Module containing functions for evaluating polynomials.

Routines in this module
-----------------------
get_poly_traces_edge(e, polys)
get_poly_vals(x, y, polys)
"""

import numpy as np

from ...mesh.edge import edge
from .poly import polynomial


def get_poly_traces_edge(e: edge, polys: list[polynomial]) -> np.ndarray:
    """
    Evaluates the polynomials in polys on the edge e.
    """
    return get_poly_vals(x=e.x[0, :], y=e.x[1, :], polys=polys)


def get_poly_vals(
    x: np.ndarray, y: np.ndarray, polys: list[polynomial]
) -> np.ndarray:
    """
    Evaluates the polynomials in polys at the points (x[j], y[j]).
    """
    num_pts = len(x)
    if len(y) != num_pts:
        raise Exception("x and y must be the same length")
    num_polys = len(polys)
    poly_vals = []
    for j in range(num_polys):
        poly_vals.append(polys[j].eval(x, y))
    return np.array(poly_vals)
