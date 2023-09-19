from numpy import array

from ...mesh.edge import edge
from .poly import polynomial


def get_poly_traces_edge(e: edge, polys: list[polynomial]):
    return get_poly_vals(x=e.x[0, :], y=e.x[1, :], polys=polys)


def get_poly_vals(x, y, polys: list[polynomial]):
    num_pts = len(x)
    if len(y) != num_pts:
        raise Exception("x and y must be the same length")
    num_polys = len(polys)
    poly_vals = []
    for j in range(num_polys):
        poly_vals.append(polys[j].eval(x, y))
    return array(poly_vals)
