"""
Wrapper for a list of Polynomials.

Classes
-------
PiecewisePolynomial
"""

from typing import Optional

from deprecated import deprecated
from numpy import ndarray, zeros

from ...mesh.cell import MeshCell
from ...mesh.edge import Edge
from .poly import Polynomial
from .poly_exceptions import PolynomialError


@deprecated(version="0.5.0", reason="Use DirichletTrace instead")
class PiecewisePolynomial:
    """
    List of Polynomials used to represent traces of vertex and Edge functions.

    Attributes
    ----------
    polys : list[Polynomial]
        List of Polynomials in the PiecewisePolynomial.
    num_polys : int
        Number of Polynomials in the PiecewisePolynomial.
    idx : int
        Identifier for the PiecewisePolynomial.

    Notes
    -----
    - This class will be deprecated in a future release. Use DirichletTrace
      instead.
    """

    polys: list[Polynomial]
    num_polys: int
    idx: int  # used to associate function with an edge or vertex

    def __init__(
        self,
        num_polys: int = 1,
        polys: Optional[list[Polynomial]] = None,
        idx: int = 0,
    ) -> None:
        """
        List of Polynomials used to represent Dirichlet traces.

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
        self.set_num_polys(num_polys)
        self.set_polys(polys)

    def set_idx(self, idx: int) -> None:
        """
        Set the identifier for the PiecewisePolynomial.

        Parameters
        ----------
        idx : int
            Identifier for the PiecewisePolynomial.
        """
        if not isinstance(idx, int):
            raise TypeError("idx must be an integer")
        if idx < 0:
            raise ValueError("idx must be nonnegative")
        self.idx = idx

    def set_num_polys(self, num_polys: int) -> None:
        """
        Set the number of Polynomials in the PiecewisePolynomial.

        Parameters
        ----------
        num_polys : int
            Number of Polynomials in the PiecewisePolynomial.
        """
        if not isinstance(num_polys, int):
            raise TypeError("num_polys must be a integer")
        if num_polys < 1:
            raise ValueError("num_polys must be positive")
        self.num_polys = num_polys

    def set_polys(self, polys: Optional[list[Polynomial]] = None) -> None:
        """
        Set the list of Polynomials in the PiecewisePolynomial.

        Parameters
        ----------
        polys : list[Polynomial], optional
            List of Polynomials in the PiecewisePolynomial. Default is None.
        """
        if polys is None:
            self.polys = [Polynomial() for _ in range(self.num_polys)]
        else:
            self.polys = polys

    def eval_on_edges(self, edges: list[Edge]) -> ndarray:
        """
        Evaluate the PiecewisePolynomial on a list of edges.

        Parameters
        ----------
        edges : list[Edge]
            List of edges on which to evaluate the PiecewisePolynomial.
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
        Evaluate the PiecewisePolynomial on the boundary of a MeshCell.

        Parameters
        ----------
        K : MeshCell
            MeshCell on which to evaluate the PiecewisePolynomial.
        """
        edges = K.get_edges()
        return self.eval_on_edges(edges)
