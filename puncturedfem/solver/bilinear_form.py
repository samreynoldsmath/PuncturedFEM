"""
Bilinear forms.

Classes
-------
BilinearForm
    Represents a bilinear form arising from a diffusion-reaction equation.
"""

from ..locfun.locfun import LocalFunction
from ..locfun.poly.poly import Polynomial
from ..locfun.trace import DirichletTrace


class BilinearForm:
    """
    Bilinear form for a diffusion-reaction equation.

    Represents a bilinear form of the form
        a(u, v) = (D grad u, grad v) + (R u, v)
    where D is the diffusion constant, R is the reaction constant. Also includes
    a right-hand side Polynomial f, so that the weak problem is
        a(u, v) = (D grad u, grad v) + (R u, v) = (f, v)
    for all v in V.

    Attributes
    ----------
    diffusion_constant : float
        Diffusion constant D
    reaction_constant : float
        Reaction constant R
    rhs_poly : Polynomial
        Right-hand side Polynomial f
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
        Initialize the BilinearForm object.

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
        """Return a string representation of the bilinear form."""
        s = "BilinearForm:"
        s += f"\n\tD: {self.diffusion_constant}"
        s += f"\n\tR: {self.reaction_constant}"
        s += f"\n\tf: {self.rhs_poly}"
        return s

    def set_diffusion_constant(self, diffusion_constant: float) -> None:
        """
        Set the diffusion constant D.

        Parameters
        ----------
        diffusion_constant : float
            The diffusion constant D
        """
        self.diffusion_constant = diffusion_constant

    def set_reaction_constant(self, reaction_constant: float) -> None:
        """
        Set the reaction constant R.

        Parameters
        ----------
        reaction_constant : float
            The reaction constant R
        """
        self.reaction_constant = reaction_constant

    def set_rhs_poly(self, f_poly: Polynomial) -> None:
        """
        Set the right-hand side polynomial f.

        Parameters
        ----------
        f_poly : Polynomial
            The right-hand side polynomial f
        """
        self.rhs_poly = f_poly

    # EVALUATION ##############################################################

    def eval_h1(self, u: LocalFunction, v: LocalFunction) -> float:
        """
        Return the H^1 semi-inner product of two LocalFunction objects u and v.

        Parameters
        ----------
        u : LocalFunction
            The first LocalFunction object
        v : LocalFunction
            The second LocalFunction object
        """
        return u.get_h1_semi_inner_prod(v)

    def eval_l2(self, u: LocalFunction, v: LocalFunction) -> float:
        """
        Return the L^2 inner product of two LocalFunction objects u and v.

        Parameters
        ----------
        u : LocalFunction
            The first LocalFunction object
        v : LocalFunction
            The second LocalFunction object
        """
        return u.get_l2_inner_prod(v)

    def eval_with_h1_and_l2(self, h1: float, l2: float) -> float:
        """
        Get the bilinear form a(u,v) given h1 and l2 inner products.

        Parameters
        ----------
        h1 : float
            The H^1 semi-inner product (grad u, grad v)
        l2 : float
            The L^2 inner product (u, v)
        """
        return self.diffusion_constant * h1 + self.reaction_constant * l2

    def eval(self, u: LocalFunction, v: LocalFunction) -> float:
        """
        Evaluate the bilinear form on two LocalFunction objects u and v.

        Parameters
        ----------
        u : LocalFunction
            The first LocalFunction object
        v : LocalFunction
            The second LocalFunction object
        """
        h1 = u.get_h1_semi_inner_prod(v)
        l2 = u.get_l2_inner_prod(v)
        return self.eval_with_h1_and_l2(h1, l2)

    def eval_rhs(self, v: LocalFunction) -> float:
        """
        Evaluate the right-hand side Polynomial f on a LocalFunction object v.

        Parameters
        ----------
        v : LocalFunction
            The LocalFunction object to evaluate f on
        """
        f = LocalFunction(
            nyst=v.nyst,
            laplacian=self.rhs_poly.laplacian(),
            trace=DirichletTrace(
                edges=v.nyst.K.get_edges(), funcs=self.rhs_poly
            ),
        )
        return f.get_l2_inner_prod(v)
