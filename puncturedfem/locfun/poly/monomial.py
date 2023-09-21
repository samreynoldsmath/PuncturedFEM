from __future__ import annotations

from typing import Optional

import numpy as np

from .multi_index import multi_index_2


class monomial:
    """
    Monomials of the form
            c (x, y) ^ (alpha_1, alpha_2) = c (x ^ alpha_1) * (y ^ alpha_2)
    """

    alpha: multi_index_2
    coef: float

    def __init__(
        self, alpha: Optional[multi_index_2] = None, coef: float = 0.0
    ) -> None:
        if alpha is None:
            alpha = multi_index_2()
        self.set_coef(coef)
        self.set_multidx(alpha)

    def copy(self) -> monomial:
        return monomial(self.alpha, self.coef)

    def is_zero(self, tol: float = 1e-12) -> bool:
        return abs(self.coef) < tol

    def set_coef(self, coef: float) -> None:
        self.coef = coef

    def set_multidx(self, alpha: multi_index_2) -> None:
        self.alpha = alpha

    def set_multidx_from_id(self, id: int) -> None:
        alpha = multi_index_2()
        alpha.set_from_id(id)
        self.set_multidx(alpha)

    def eval(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        val = self.coef * np.ones(np.shape(x))
        if self.alpha.x > 0:
            val *= x**self.alpha.x
        if self.alpha.y > 0:
            val *= y**self.alpha.y
        return val

    def partial_deriv(self, var: str) -> monomial:
        if var == "x":
            if self.alpha.x == 0:
                # constant wrt x
                b = 0.0
                beta = multi_index_2()
            else:
                # power rule
                b = self.coef * self.alpha.x
                beta = multi_index_2([self.alpha.x - 1, self.alpha.y])
        elif var == "y":
            if self.alpha.y == 0:
                # constant wrt y
                b = 0.0
                beta = multi_index_2()
            else:
                # power rule
                b = self.coef * self.alpha.y
                beta = multi_index_2([self.alpha.x, self.alpha.y - 1])
        else:
            raise Exception('var must be one of the strings "x" or "y"')
        return monomial(alpha=beta, coef=b)

    def grad(self) -> tuple[monomial, monomial]:
        gx = self.partial_deriv("x")
        gy = self.partial_deriv("y")
        return gx, gy

    def __repr__(self) -> str:
        # coefficient
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
        if not isinstance(other, monomial):
            raise TypeError("Comparison of monomial to non-monomial object")
        same_id = self.alpha.id == other.alpha.id
        same_coef = abs(self.coef - other.coef) < tol
        return same_id and same_coef

    def __gt__(self, other: object) -> bool:
        """
        Returns True iff self.id > other.id
        """
        if not isinstance(other, monomial):
            raise TypeError("Comparison of monomial to non-monomial object")
        return self.alpha.id > other.alpha.id

    def __add__(self, other: object) -> monomial:
        if not isinstance(other, monomial):
            raise TypeError("Cannot add monomial to non-monomial object")
        if not self.alpha == other.alpha:
            raise ValueError(
                "Cannot add monomomials with different mulit-"
                + "indices. Use a polynomial object instead"
            )
        return monomial(self.alpha, self.coef + other.coef)

    def __mul__(self, other: object) -> monomial:
        """
        Defines the operation self * other
        where other is either a monomial object or a scalar
        """
        if isinstance(other, monomial):
            # multiplication between two monomials
            b = self.coef * other.coef
            beta = self.alpha + other.alpha
            return monomial(beta, b)
        if isinstance(other, (int, float)):
            # scalar multiplication
            b = self.coef * other
            beta = self.alpha.copy()
            return monomial(beta, b)
        raise TypeError(
            "Multiplication by monomial must be by a scalar or"
            + " by another monomial"
        )

    def __rmul__(self, other: object) -> monomial:
        """
        Defines the operation: other * self
        where other is either a monomial object or a scalar
        """
        if isinstance(other, (int, float)):
            return self * other
        raise TypeError(
            "Multiplication by monomial must be by a scalar or"
            + " by another monomial"
        )

    def __neg__(self) -> None:
        """
        Defines negation operation: -self
        """
        self.coef *= -1
