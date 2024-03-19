"""
Monomial.py
===========

Module containing the Monomial class, which is used to represent Monomials
of the form
    m(x) = c * x^alpha
         = c * (x_1, x_2) ^ (alpha_1, alpha_2)
         = c * (x_1 ^ alpha_1) * (x_2 ^ alpha_2)
where alpha = (alpha_1, alpha_2) is a multi-index and c is a scalar.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...util.types import FloatLike
from .multi_index import MultiIndex
from .poly_exceptions import InvalidVariableError


class Monomial:
    """
    Monomials of the form
            c (x, y) ^ (alpha_1, alpha_2) = c (x ^ alpha_1) * (y ^ alpha_2)
    """

    alpha: MultiIndex
    coef: float

    def __init__(
        self, alpha: Optional[MultiIndex] = None, coef: float = 0.0
    ) -> None:
        """
        Constructor for Monomial class.

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
        Returns a copy of self.
        """
        return Monomial(self.alpha, self.coef)

    def is_zero(self, tol: float = 1e-12) -> bool:
        """
        Returns True iff self is the zero Monomial. Default tolerance is 1e-12.
        """
        return abs(self.coef) < tol

    def set_coef(self, coef: float) -> None:
        """
        Set the coefficient of the Monomial to coef.
        """
        self.coef = coef

    def set_multidx(self, alpha: MultiIndex) -> None:
        """
        Set the multi-index of the Monomial to alpha.
        """
        self.alpha = alpha

    def set_multidx_from_idx(self, idx: int) -> None:
        """
        Set the multi-index of the Monomial to the multi-index with id = id.
        """
        alpha = MultiIndex()
        alpha.set_from_idx(idx)
        self.set_multidx(alpha)

    def eval(self, x: FloatLike, y: FloatLike) -> np.ndarray:
        """
        Evaluate the Monomial at the point (x, y).
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
        """
        gx = self.partial_deriv("x")
        gy = self.partial_deriv("y")
        return gx, gy

    def __repr__(self) -> str:
        """
        Returns a string representation of self.
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
        Returns True iff self == other
        """
        if not isinstance(other, Monomial):
            raise TypeError("Comparison of Monomial to non-Monomial object")
        if self.alpha.idx != other.alpha.idx:
            return False
        return abs(self.coef - other.coef) < tol

    def __gt__(self, other: object) -> bool:
        """
        Returns True iff self.idx > other.idx
        """
        if not isinstance(other, Monomial):
            raise TypeError("Comparison of Monomial to non-Monomial object")
        return self.alpha.idx > other.alpha.idx

    def __add__(self, other: object) -> Monomial:
        """
        Defines the operation self + other
        where other is either a Monomial object or a scalar
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
        Defines the operation self * other
        where other is either a Monomial object or a scalar
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
        Defines the operation: other * self
        where other is either a Monomial object or a scalar
        """
        if isinstance(other, (int, float)):
            return self * other
        raise TypeError(
            "Multiplication by Monomial must be by a scalar or"
            + " by another Monomial"
        )

    def __neg__(self) -> None:
        """
        Defines negation operation: -self
        """
        self.coef *= -1
