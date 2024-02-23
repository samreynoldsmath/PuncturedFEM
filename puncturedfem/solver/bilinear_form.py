"""
BilinearForm.py
================

Module containing the BilinearForm class, which is used to represent a bilinear
form.
"""

from ..locfun.locfun import LocalFunction
from ..locfun.poly.piecewise_poly import PiecewisePolynomial
from ..locfun.poly.poly import Polynomial


class BilinearForm:
    """
    Represents a bilinear form of the form
        a(u, v) = (D grad u, grad v) + (R u, v)
    where D is the diffusion constant, R is the reaction constant. Also includes
    a right-hand side Polynomial f, so that the weak problem is
        a(u, v) = (D grad u, grad v) + (R u, v) = (f, v)
    for all v in V.
    """

    diffusion_constant: float
    reaction_constant: float
    rhs_poly: Polynomial

    def __init__(
        self,
        diffusion_constant: float,
        reaction_constant: float,
        rhs_poly: Polynomial,
    ) -> None:
        """
        Constructor for BilinearForm class.

        Parameters
        ----------
        diffusion_constant : float
            Diffusion constant D
        reaction_constant : float
            Reaction constant R
        rhs_poly : Polynomial
            Right-hand side Polynomial f
        """
        self.set_diffusion_constant(diffusion_constant)
        self.set_reaction_constant(reaction_constant)
        self.set_rhs_poly(rhs_poly)

    def __str__(self) -> str:
        """
        Returns a string representation of the bilinear form.
        """
        s = "BilinearForm:"
        s += f"\n\tD: {self.diffusion_constant}"
        s += f"\n\tR: {self.reaction_constant}"
        s += f"\n\tf: {self.rhs_poly}"
        return s

    def set_diffusion_constant(self, diffusion_constant: float) -> None:
        """
        Sets the diffusion constant D.
        """
        self.diffusion_constant = diffusion_constant

    def set_reaction_constant(self, reaction_constant: float) -> None:
        """
        Sets the reaction constant R.
        """
        self.reaction_constant = reaction_constant

    def set_rhs_poly(self, f_poly: Polynomial) -> None:
        """
        Sets the right-hand side Polynomial f.
        """
        self.rhs_poly = f_poly

    # EVALUATION ##############################################################

    def eval_h1(self, u: LocalFunction, v: LocalFunction) -> float:
        """
        Returns the H^1 semi-inner product of twoLocalFunction objects u and v.
        """
        return u.get_h1_semi_inner_prod(v)

    def eval_l2(self, u: LocalFunction, v: LocalFunction) -> float:
        """
        Returns the L^2 inner product of twoLocalFunction objects u and v.
        """
        return u.get_l2_inner_prod(v)

    def eval_with_h1_and_l2(self, h1: float, l2: float) -> float:
        """
        Returns the bilinear form a(u,v) given the h1 and l2 (semi-)inner
        products of u and v.
        """
        return self.diffusion_constant * h1 + self.reaction_constant * l2

    def eval(self, u: LocalFunction, v: LocalFunction) -> float:
        """
        Evaluates the bilinear form on twoLocalFunction objects u and v.
        """
        h1 = u.get_h1_semi_inner_prod(v)
        l2 = u.get_l2_inner_prod(v)
        return self.eval_with_h1_and_l2(h1, l2)

    def eval_rhs(self, v: LocalFunction) -> float:
        """
        Evaluates the right-hand side Polynomial f on aLocalFunction object v.
        """
        m = len(v.poly_trace.polys)
        poly_trace = PiecewisePolynomial(
            num_polys=m,
            polys=[self.rhs_poly for _ in range(len(v.poly_trace.polys))],
        )
        f = LocalFunction(
            nyst=v.nyst,
            lap_poly=self.rhs_poly.laplacian(),
            poly_trace=poly_trace,
        )
        f.compute_all()
        return f.get_l2_inner_prod(v)
