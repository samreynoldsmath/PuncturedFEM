"""
Monomials in two variables.

Classes
-------
Monomial
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...util.types import FloatLike
from .multi_index import MultiIndex
from .poly_exceptions import InvalidVariableError


class Monomial:
    """
    Monomial of the form c * x_1 ^ alpha_1 * x_2 ^ alpha_2.

    Attributes
    ----------
    alpha : MultiIndex
        Multi-index of the Monomial.
    coef : float
        Coefficient of the Monomial.
    """

    alpha: MultiIndex
    coef: float

    def __init__(
        self, alpha: Optional[MultiIndex] = None, coef: float = 0.0
    ) -> None:
        """
        Monomial of the form c * x_1 ^ alpha_1 * x_2 ^ alpha_2.

        Parameters
        ----------
        alpha : MultiIndex, optional
            Multi-index of the Monomial. Default is None.
        coef : float, optional
            Coefficient of the Monomial. Default is 0.0.
        """
        if alpha is None:
            alpha = MultiIndex()
        self.set_coef(coef)
        self.set_multidx(alpha)

    def copy(self) -> Monomial:
        """
        Return a copy of self.

        Returns
        -------
        Monomial
            A copy of self.
        """
        return Monomial(self.alpha, self.coef)

    def is_zero(self, tol: float = 1e-12) -> bool:
        """
        Return True iff self is the zero Monomial.

        Parameters
        ----------
        tol : float, optional
            Tolerance for the comparison. Default is 1e-12.

        Returns
        -------
        bool
            True iff self is the zero Monomial.

        Notes
        -----
        This method is deprecated and will be removed in a future release.
        """
        return abs(self.coef) < tol

    def set_coef(self, coef: float) -> None:
        """
        Set the coefficient of the Monomial to coef.

        Parameters
        ----------
        coef : float
            Coefficient of the Monomial.
        """
        self.coef = coef

    def set_multidx(self, alpha: MultiIndex) -> None:
        """
        Set the multi-index of the Monomial to alpha.

        Parameters
        ----------
        alpha : MultiIndex
            Multi-index of the Monomial.

        See Also
        --------
        MultiIndex: Class representing a multi-index.
        """
        self.alpha = alpha

    def set_multidx_from_idx(self, idx: int) -> None:
        """
        Set the multi-index via lexical ordering using the index idx.

        Parameters
        ----------
        idx : int
            Lexical ordering index of the multi-index.

        See Also
        --------
        MultiIndex: Class representing a multi-index.
        """
        alpha = MultiIndex()
        alpha.set_from_idx(idx)
        self.set_multidx(alpha)

    def eval(self, x: FloatLike, y: FloatLike) -> np.ndarray:
        """
        Evaluate the Monomial at the point (x, y).

        Parameters
        ----------
        x : FloatLike
            x-coordinate of the point at which to evaluate the Monomial.
        y : FloatLike
            y-coordinate of the point at which to evaluate the Monomial.

        Returns
        -------
        np.ndarray
            Value of the Monomial at the point (x, y).
        """
        val = self.coef * np.ones(np.shape(x))
        if self.alpha.x > 0:
            val *= x**self.alpha.x
        if self.alpha.y > 0:
            val *= y**self.alpha.y
        return val

    def partial_deriv(self, var: str) -> Monomial:
        """
        Compute the partial derivative of self with respect to var.

        Parameters
        ----------
        var : str
            The variable with respect to which to differentiate. Must be
            one of the strings "x" or "y".

        Returns
        -------
        Monomial
            The partial derivative of self with respect to var.
        """
        if var == "x":
            if self.alpha.x == 0:
                # constant wrt x
                b = 0.0
                beta = MultiIndex()
            else:
                # power rule
                b = self.coef * self.alpha.x
                beta = MultiIndex([self.alpha.x - 1, self.alpha.y])
        elif var == "y":
            if self.alpha.y == 0:
                # constant wrt y
                b = 0.0
                beta = MultiIndex()
            else:
                # power rule
                b = self.coef * self.alpha.y
                beta = MultiIndex([self.alpha.x, self.alpha.y - 1])
        else:
            raise InvalidVariableError(
                'var must be one of the strings "x" or "y"'
            )
        return Monomial(alpha=beta, coef=b)

    def grad(self) -> tuple[Monomial, Monomial]:
        """
        Compute the gradient of self.

        Returns
        -------
        tuple[Monomial, Monomial]
            The partial derivatives of self with respect to x and y.
        """
        gx = self.partial_deriv("x")
        gy = self.partial_deriv("y")
        return gx, gy

    def __repr__(self) -> str:
        """
        Return a string representation of self.

        Returns
        -------
        str
            String representation of self.
        """
        msg = f"+ ({self.coef}) "

        # power of x
        if self.alpha.x > 0:
            msg += "x"
        if self.alpha.x > 1:
            msg += f"^{self.alpha.x} "
        else:
            msg += " "

        # power of y
        if self.alpha.y > 0:
            msg += "y"
        if self.alpha.y > 1:
            msg += f"^{self.alpha.y} "
        else:
            msg += " "

        return msg

    def __eq__(self, other: object, tol: float = 1e-12) -> bool:
        """
        Return True iff self == other.

        Parameters
        ----------
        other : object
            The object to compare to self. Must be another Monomial.

        Returns
        -------
        bool
            True iff self == other.
        """
        if not isinstance(other, Monomial):
            raise TypeError("Comparison of Monomial to non-Monomial object")
        if self.alpha.idx != other.alpha.idx:
            return False
        return abs(self.coef - other.coef) < tol

    def __gt__(self, other: object) -> bool:
        """
        Return True iff self.idx > other.idx.

        Parameters
        ----------
        other : object
            The object to compare to self. Must be another Monomial.

        Returns
        -------
        bool
            True iff self.idx > other.idx.
        """
        if not isinstance(other, Monomial):
            raise TypeError("Comparison of Monomial to non-Monomial object")
        return self.alpha.idx > other.alpha.idx

    def __add__(self, other: object) -> Monomial:
        """
        Define the operation self + other.

        Parameters
        ----------
        other : object
            The object to add to self. Must be another Monomial.

        Returns
        -------
        Monomial
            The result of the addition.
        """
        if not isinstance(other, Monomial):
            raise TypeError("Cannot add Monomial to non-Monomial object")
        if not self.alpha == other.alpha:
            raise ValueError(
                "Cannot add Monomials with different multi-"
                + "indices. Use a Polynomial object instead"
            )
        return Monomial(self.alpha, self.coef + other.coef)

    def __mul__(self, other: object) -> Monomial:
        """
        Define the operation self * other.

        Parameters
        ----------
        other : object
            The object to multiply by self. Must be a scalar (int or float)
            or another Monomial.

        Returns
        -------
        Monomial
            The result of the multiplication.
        """
        if isinstance(other, Monomial):
            # multiplication between two Monomials
            b = self.coef * other.coef
            beta = self.alpha + other.alpha
            return Monomial(beta, b)
        if isinstance(other, (int, float)):
            # scalar multiplication
            b = self.coef * other
            beta = self.alpha.copy()
            return Monomial(beta, b)
        raise TypeError(
            "Multiplication by Monomial must be by a scalar or"
            + " by another Monomial"
        )

    def __rmul__(self, other: object) -> Monomial:
        """
        Define the operation: other * self.

        Parameters
        ----------
        other : object
            The object to multiply by self. Must be a scalar (int or float).

        Returns
        -------
        Monomial
            The result of the multiplication.
        """
        if isinstance(other, (int, float)):
            return self * other
        raise TypeError(
            "Multiplication by Monomial must be by a scalar or"
            + " by another Monomial"
        )

    def __neg__(self) -> None:
        """Define negation operation: -self."""
        self.coef *= -1
