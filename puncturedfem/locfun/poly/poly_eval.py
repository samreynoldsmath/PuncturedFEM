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
from deprecated import deprecated

from ...mesh.edge import Edge
from ...mesh.mesh_exceptions import SizeMismatchError
from .poly import Polynomial


def get_poly_traces_edge(e: Edge, polys: list[Polynomial]) -> np.ndarray:
    """
    Evaluates the Polynomials in polys on the Edge e.
    """
    return get_poly_vals(x=e.x[0, :], y=e.x[1, :], polys=polys)


@deprecated(version="0.4.3", reason="Use direct call instead")
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
        poly_vals.append(polys[j](x, y))
    return np.array(poly_vals)
