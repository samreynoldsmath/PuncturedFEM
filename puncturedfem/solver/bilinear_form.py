"""
bilinear_form.py
================

Module containing the bilinear_form class, which is used to represent a bilinear
form.
"""

from ..locfun.locfun import locfun
from ..locfun.poly.piecewise_poly import piecewise_polynomial
from ..locfun.poly.poly import polynomial


class bilinear_form:
    """
    Represents a bilinear form of the form
        a(u, v) = (D grad u, grad v) + (R u, v)
    where D is the diffusion constant, R is the reaction constant. Also includes
    a right-hand side polynomial f, so that the weak problem is
        a(u, v) = (D grad u, grad v) + (R u, v) = (f, v)
    for all v in V.
    """

    diffusion_constant: float
    reaction_constant: float
    rhs_poly: polynomial

    def __init__(
        self,
        diffusion_constant: float,
        reaction_constant: float,
        rhs_poly: polynomial,
    ) -> None:
        """
        Constructor for bilinear_form class.

        Parameters
        ----------
        diffusion_constant : float
            Diffusion constant D
        reaction_constant : float
            Reaction constant R
        rhs_poly : polynomial
            Right-hand side polynomial f
        """
        self.set_diffusion_constant(diffusion_constant)
        self.set_reaction_constant(reaction_constant)
        self.set_rhs_poly(rhs_poly)

    def __str__(self) -> str:
        """
        Returns a string representation of the bilinear form.
        """
        s = "bilinear_form:"
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

    def set_rhs_poly(self, f_poly: polynomial) -> None:
        """
        Sets the right-hand side polynomial f.
        """
        self.rhs_poly = f_poly

    def eval(self, u: locfun, v: locfun) -> float:
        """
        Evaluates the bilinear form on two locfun objects u and v.
        """
        h1 = u.get_h1_semi_inner_prod(v)
        l2 = u.get_l2_inner_prod(v)
        return self.diffusion_constant * h1 + self.reaction_constant * l2

    def eval_rhs(self, v: locfun) -> float:
        """
        Evaluates the right-hand side polynomial f on a locfun object v.
        """
        m = len(v.poly_trace.polys)
        poly_trace = piecewise_polynomial(
            num_polys=m,
            polys=[self.rhs_poly for _ in range(len(v.poly_trace.polys))],
        )
        f = locfun(
            v.solver, lap_poly=self.rhs_poly.laplacian(), poly_trace=poly_trace
        )
        f.compute_all()
        return f.get_l2_inner_prod(v)
