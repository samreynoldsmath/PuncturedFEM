"""
piecewise_poly.py
=================

Module containing the piecewise_polynomial class. It is essentially a wrapper
for a list of polynomials.
"""

from typing import Optional

from numpy import ndarray, zeros

from ...mesh.cell import cell
from ...mesh.edge import edge
from .poly import polynomial
from .poly_exceptions import PolynomialError


class piecewise_polynomial:
    """List of polynomials used to represent traces of vertex and edge funs"""

    polys: list[polynomial]
    num_polys: int
    id: int  # used to associate function with an edge or vertex

    def __init__(
        self,
        num_polys: int = 1,
        polys: Optional[list[polynomial]] = None,
        idx: int = 0,
    ) -> None:
        """
        Constructor for piecewise_polynomial class.

        Parameters
        ----------
        num_polys : int, optional
            Number of polynomials in the piecewise_polynomial. Default is 1.
        polys : list[polynomial], optional
            List of polynomials in the piecewise_polynomial. Default is None.
        idx : int, optional
            Identifier for the piecewise_polynomial. Default is 0.
        """
        self.set_idx(idx)
        self.set_num_polys(num_polys)  # TODO: determine num_polys automatically
        self.set_polys(polys)

    def set_idx(self, idx: int) -> None:
        """
        Sets the identifier for the piecewise_polynomial.
        """
        if not isinstance(idx, int):
            raise TypeError("idx must be an integer")
        if idx < 0:
            raise ValueError("idx must be nonnegative")
        self.idx = idx

    def set_num_polys(self, num_polys: int) -> None:
        """
        Sets the number of polynomials in the piecewise_polynomial.
        """
        if not isinstance(num_polys, int):
            raise TypeError("num_polys must be a integer")
        if num_polys < 1:
            raise ValueError("num_polys must be positive")
        self.num_polys = num_polys

    def set_polys(self, polys: Optional[list[polynomial]] = None) -> None:
        """
        Sets the list of polynomials in the piecewise_polynomial.
        """
        if polys is None:
            self.polys = [polynomial() for _ in range(self.num_polys)]
        else:
            self.polys = polys

    def eval_on_edges(self, edges: list[edge]) -> ndarray:
        """
        Evaluates the piecewise_polynomial on a list of edges.
        """
        m = len(edges)
        if m != self.num_polys:
            raise PolynomialError(
                "Number of edges must match number of polynomials"
            )
        vals_arr = []
        num_pts = 0
        for i in range(m):
            e = edges[i]
            num_pts += e.num_pts
            vals_arr.append(self.polys[i].eval(x=e.x[0, :], y=e.x[1, :]))
        vals = zeros((num_pts,))
        idx = 0
        for i in range(m):
            e = edges[i]
            vals[idx : idx + e.num_pts] = vals_arr[i]
        return vals

    def eval_on_cell_boundary(self, K: cell) -> ndarray:
        """
        Evaluates the piecewise_polynomial on the boundary of a cell.
        """
        edges = K.get_edges()
        return self.eval_on_edges(edges)
