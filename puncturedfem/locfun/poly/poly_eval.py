import numpy as np

from ...mesh.edge import edge
from .poly import polynomial


def get_poly_traces_edge(e: edge, polys: list[polynomial]) -> np.ndarray:
    return get_poly_vals(x=e.x[0, :], y=e.x[1, :], polys=polys)


def get_poly_vals(
    x: np.ndarray, y: np.ndarray, polys: list[polynomial]
) -> np.ndarray:
    num_pts = len(x)
    if len(y) != num_pts:
        raise Exception("x and y must be the same length")
    num_polys = len(polys)
    poly_vals = []
    for j in range(num_polys):
        poly_vals.append(polys[j].eval(x, y))
    return np.array(poly_vals)
