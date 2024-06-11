"""
Polynomial on a mesh cell.

Classes
-------
LocalPolynomial
    Polynomial on a mesh cell.
"""

from __future__ import annotations
from typing import Optional, Union

from ..mesh.cell import MeshCell
from .poly.poly import Polynomial
from .trace import DirichletTrace


class LocalPolynomial:
    """
    Polynomial on a mesh cell.

    A typical use case for this class is to represent a polynomial part of a
    local Poisson function.

    The weighted normal derivative of the polynomial and that of its
    anti-Laplacian are computed and stored in their respective DirichletTrace
    objects.

    Attributes
    ----------
    exact_form : Polynomial
        Exact form of the polynomial.
    trace : DirichletTrace
        Dirichlet trace of the polynomial.
    grad1 : Polynomial
        First gradient of the polynomial.
    grad2 : Polynomial
        Second gradient of the polynomial.
    antilap : Polynomial
        Anti-Laplacian of the polynomial.
    antilap_trace : DirichletTrace
        Dirichlet trace of the anti-Laplacian.
    """

    exact_form: Polynomial
    trace: DirichletTrace
    grad1: Polynomial
    grad2: Polynomial
    antilap: Polynomial
    antilap_trace: DirichletTrace

    def __init__(
        self, exact_form: Polynomial, mesh_cell: Optional[MeshCell]
    ) -> None:
        """
        Initialize the polynomial.

        Parameters
        ----------
        exact_form : Polynomial
            Exact form of the polynomial.
        mesh_cell : MeshCell
            The mesh cell on which the polynomial is defined.
        """
        self.exact_form = exact_form
        if mesh_cell is None:
            return
        self.trace = DirichletTrace(
            edges=mesh_cell.get_edges(),
            values=exact_form(*mesh_cell.get_boundary_points()),
        )
        self.grad1, self.grad2 = exact_form.grad()
        self.trace.set_weighted_normal_derivative(
            exact_form.get_weighted_normal_derivative(mesh_cell)
        )
        self.trace.set_weighted_tangential_derivative(
            exact_form.get_weighted_tangential_derivative(mesh_cell)
        )
        self.antilap = exact_form.anti_laplacian()
        self.antilap_trace = DirichletTrace(
            edges=mesh_cell.get_edges(),
            values=self.antilap(*mesh_cell.get_boundary_points()),
        )
        self.antilap_trace.set_weighted_normal_derivative(
            self.antilap.get_weighted_normal_derivative(mesh_cell)
        )

    def __add__(self, other: LocalPolynomial) -> LocalPolynomial:
        """
        Add two local polynomials.

        Parameters
        ----------
        other : LocalPolynomial
            The other local polynomial.

        Returns
        -------
        LocalPolynomial
            The sum of the two local polynomials.
        """
        if not isinstance(other, LocalPolynomial):
            raise TypeError("The other term must be a LocalPolynomial.")
        new = LocalPolynomial(
            exact_form=self.exact_form + other.exact_form, mesh_cell=None
        )
        new.trace = self.trace + other.trace
        new.grad1 = self.grad1 + other.grad1
        new.grad2 = self.grad2 + other.grad2
        new.antilap = self.antilap + other.antilap
        new.antilap_trace = self.antilap_trace + other.antilap_trace
        return new

    def __mul__(self, other: Union[int, float]) -> LocalPolynomial:
        """
        Multiply a local polynomial by a scalar.

        Parameters
        ----------
        other : Union[int, float]
            The scalar.

        Returns
        -------
        LocalPolynomial
            The product of the local polynomial and the scalar.
        """
        if not isinstance(other, (int, float)):
            raise TypeError("The multiplier must be a scalar.")
        new = LocalPolynomial(
            exact_form=self.exact_form * other, mesh_cell=None
        )
        new.trace = self.trace * other
        new.grad1 = self.grad1 * other
        new.grad2 = self.grad2 * other
        new.antilap = self.antilap * other
        new.antilap_trace = self.antilap_trace * other
        return new

    def __rmul__(self, other: Union[int, float]) -> LocalPolynomial:
        """
        Multiply a local polynomial by a scalar.

        Parameters
        ----------
        other : Union[int, float]
            The scalar.

        Returns
        -------
        LocalPolynomial
            The product of the local polynomial and the scalar.
        """
        return self.__mul__(other)

    def __truediv__(self, other: Union[int, float]) -> LocalPolynomial:
        """
        Divide a local polynomial by a scalar.

        Parameters
        ----------
        other : Union[int, float]
            The scalar.

        Returns
        -------
        LocalPolynomial
            The division of the local polynomial by the scalar.
        """
        if not isinstance(other, (int, float)):
            raise TypeError("The divisor must be a scalar.")
        if other == 0:
            raise ValueError("Division by zero.")
        new = LocalPolynomial(
            exact_form=self.exact_form / other, mesh_cell=None
        )
        new.trace = self.trace / other
        new.grad1 = self.grad1 / other
        new.grad2 = self.grad2 / other
        new.antilap = self.antilap / other
        new.antilap_trace = self.antilap_trace / other
        return new

    def __sub__(self, other: LocalPolynomial) -> LocalPolynomial:
        """
        Subtract two local polynomials.

        Parameters
        ----------
        other : LocalPolynomial
            The other local polynomial.

        Returns
        -------
        LocalPolynomial
            The difference of the two local polynomials.
        """
        if not isinstance(other, LocalPolynomial):
            raise TypeError("The other term must be a LocalPolynomial.")
        new = LocalPolynomial(
            exact_form=self.exact_form - other.exact_form, mesh_cell=None
        )
        new.trace = self.trace - other.trace
        new.grad1 = self.grad1 - other.grad1
        new.grad2 = self.grad2 - other.grad2
        new.antilap = self.antilap - other.antilap
        new.antilap_trace = self.antilap_trace - other.antilap_trace
        return new
