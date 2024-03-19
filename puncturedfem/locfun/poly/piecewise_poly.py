"""
piecewise_poly.py
=================

Module containing the PiecewisePolynomial class. It is essentially a wrapper
for a list of Polynomials.

TODO: Deprecate the PiecewisePolynomial class
"""

from typing import Optional

# from deprecated import deprecated
from numpy import ndarray, zeros

from ...mesh.cell import MeshCell
from ...mesh.edge import Edge
from .poly import Polynomial
from .poly_exceptions import PolynomialError


class PiecewisePolynomial:
    """List of Polynomials used to represent traces of vertex and Edge funs"""

    polys: list[Polynomial]
    num_polys: int
    id: int  # used to associate function with an Edge or vertex

    # TODO: deprecated in favor of DirichletTrace
    # @deprecated(version="0.5.0", reason="Use DirichletTrace instead")
    def __init__(
        self,
        num_polys: int = 1,
        polys: Optional[list[Polynomial]] = None,
        idx: int = 0,
    ) -> None:
        """
        Constructor for PiecewisePolynomial class.

        Parameters
        ----------
        num_polys : int, optional
            Number of Polynomials in the PiecewisePolynomial. Default is 1.
        polys : list[Polynomial], optional
            List of Polynomials in the PiecewisePolynomial. Default is None.
        idx : int, optional
            Identifier for the PiecewisePolynomial. Default is 0.
        """
        self.set_idx(idx)
        self.set_num_polys(num_polys)  # TODO: determine num_polys automatically
        self.set_polys(polys)

    def set_idx(self, idx: int) -> None:
        """
        Sets the identifier for the PiecewisePolynomial.
        """
        if not isinstance(idx, int):
            raise TypeError("idx must be an integer")
        if idx < 0:
            raise ValueError("idx must be nonnegative")
        self.idx = idx

    def set_num_polys(self, num_polys: int) -> None:
        """
        Sets the number of Polynomials in the PiecewisePolynomial.
        """
        if not isinstance(num_polys, int):
            raise TypeError("num_polys must be a integer")
        if num_polys < 1:
            raise ValueError("num_polys must be positive")
        self.num_polys = num_polys

    def set_polys(self, polys: Optional[list[Polynomial]] = None) -> None:
        """
        Sets the list of Polynomials in the PiecewisePolynomial.
        """
        if polys is None:
            self.polys = [Polynomial() for _ in range(self.num_polys)]
        else:
            self.polys = polys

    def eval_on_edges(self, edges: list[Edge]) -> ndarray:
        """
        Evaluates the PiecewisePolynomial on a list of edges.
        """
        m = len(edges)
        if m != self.num_polys:
            raise PolynomialError(
                "Number of edges must match number of Polynomials"
            )
        vals_arr = []
        num_pts = 0
        for i in range(m):
            e = edges[i]
            num_pts += e.num_pts
            vals_arr.append(self.polys[i](x=e.x[0, :], y=e.x[1, :]))
        vals = zeros((num_pts,))
        idx = 0
        for i in range(m):
            e = edges[i]
            vals[idx : idx + e.num_pts] = vals_arr[i]
        return vals

    def eval_on_mesh_boundary(self, K: MeshCell) -> ndarray:
        """
        Evaluates the PiecewisePolynomial on the boundary of a MeshCell.
        """
        edges = K.get_edges()
        return self.eval_on_edges(edges)
