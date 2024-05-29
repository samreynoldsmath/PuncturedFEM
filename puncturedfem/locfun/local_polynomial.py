"""
Polynomial on a mesh cell.

Classes
-------
LocalPolynomial
    Polynomial on a mesh cell.
"""

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

    def __init__(self, exact_form: Polynomial, K: MeshCell) -> None:
        """
        Initialize the polynomial.

        Parameters
        ----------
        exact_form : Polynomial
            Exact form of the polynomial.
        K : MeshCell
            The mesh cell on which the polynomial is defined.
        """
        self.exact_form = exact_form
        self.trace = DirichletTrace(
            edges=K.get_edges(), values=exact_form(*K.get_boundary_points())
        )
        self.grad1, self.grad2 = exact_form.grad()
        self.trace.set_weighted_normal_derivative(
            exact_form.get_weighted_normal_derivative(K)
        )
        self.trace.set_weighted_tangential_derivative(
            exact_form.get_weighted_tangential_derivative(K)
        )
        self.antilap = exact_form.anti_laplacian()
        self.antilap_trace = DirichletTrace(
            edges=K.get_edges(), values=self.antilap(*K.get_boundary_points())
        )
        self.antilap_trace.set_weighted_normal_derivative(
            self.antilap.get_weighted_normal_derivative(K)
        )
