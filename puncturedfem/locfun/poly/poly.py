"""
Polynomial in two variables.

Classes
-------
Polynomial
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from deprecated import deprecated

from ...mesh.cell import MeshCell
from ...util.types import FloatLike
from .monomial import Monomial
from .multi_index import MultiIndex
from .poly_exceptions import MultiIndexError, PolynomialError


class Polynomial:
    """
    Polynomial in two variables, represented as a list of Monomials.

    Attributes
    ----------
    monos : list[Monomial]
        List of Monomials in the Polynomial.
    """

    monos: list[Monomial]

    def __init__(
        self, coef_multidx_pairs: Optional[list[tuple[float, int, int]]] = None
    ) -> None:
        """
        Polynomial in two variables, represented as a list of Monomials.

        Parameters
        ----------
        coef_multidx_pairs : list[tuple[float, int, int]], optional
            List of coefficient / multi-index pairs, by default None
        """
        self.set(coef_multidx_pairs)

    def set(
        self, coef_multidx_pairs: Optional[list[tuple[float, int, int]]] = None
    ) -> None:
        """
        Set the Polynomial to the list of coefficient / multi-index pairs.

        Parameters
        ----------
        coef_multidx_pairs : list[tuple[float, int, int]], optional
            List of coefficient / multi-index pairs, by default None
        """
        self.monos = []
        if coef_multidx_pairs is None:
            return
        for triple in coef_multidx_pairs:
            if len(triple) != 3:
                raise MultiIndexError(
                    "Every multi-index / coefficient pair must consist of"
                    + "\n\t[0]:\tthe coefficient"
                    + "\n\t[1]:\tthe exponent on x_1"
                    + "\n\t[2]:\tthe exponent on x_2"
                )
            c = float(triple[0])
            alpha = MultiIndex([triple[1], triple[2]])
            m = Monomial(alpha, c)
            self.add_monomial(m)
        self.consolidate()

    def copy(self) -> Polynomial:
        """
        Get a copy of the Polynomial.

        Returns
        -------
        Polynomial
            A copy of the Polynomial.
        """
        new = Polynomial()
        new.add_monomials(self.monos)
        return new

    def add_monomial(self, m: Monomial) -> None:
        """
        Add a Monomial to the Polynomial.

        Parameters
        ----------
        m : Monomial
            Monomial to add to the Polynomial.
        """
        if not m.is_zero():
            self.monos.append(m)

    def add_monomials(self, monos: Optional[list[Monomial]] = None) -> None:
        """
        Add a list of Monomials to the Polynomial.

        Parameters
        ----------
        monos : list[Monomial], optional
            List of Monomials to add to the Polynomial, by default None
        """
        if monos is None:
            return
        for m in monos:
            self.add_monomial(m)
        self.consolidate()

    def remove_zeros(self) -> None:
        """Remove terms with zero coefficients."""
        for i in range(len(self.monos), 0, -1):
            if self.monos[i - 1].is_zero():
                del self.monos[i - 1]

    def consolidate(self) -> None:
        """Consolidate the coefficients of repeated indices."""
        N = len(self.monos)
        for i in range(N):
            for j in range(i + 1, N):
                if self.monos[i].alpha == self.monos[j].alpha:
                    self.monos[i] += self.monos[j]
                    self.monos[j] *= 0
        self.remove_zeros()
        self.sort()

    def sort(self) -> None:
        """
        Sort the Monomials according to multi-index id.

        Notes
        -----
        Using Insertion Sort algorithm since Monomial list is assumed be be
        short.
        """
        for i in range(len(self.monos)):
            j = i
            while j > 0 and self.monos[j - 1] > self.monos[j]:
                temp = self.monos[j - 1]
                self.monos[j - 1] = self.monos[j]
                self.monos[j] = temp
                j -= 1

    def add_monomial_with_idx(self, coef: float, idx: int) -> None:
        """
        Add a Monomial with a given lexicographical index idx.

        Parameters
        ----------
        coef : float
            Coefficient of the Monomial.
        idx : int
            Lexicographical index of the Monomial.

        See Also
        --------
        MultiIndex
        """
        m = Monomial()
        m.set_multidx_from_idx(idx)
        m.set_coef(coef)
        self.add_monomial(m)
        self.consolidate()

    def add_monomials_with_idxs(
        self, coef_list: list[float], idx_list: list[int]
    ) -> None:
        """
        Add list of Monomials with given lexicographical indices.

        Parameters
        ----------
        coef_list : list[float]
            List of coefficients of the Monomials.
        idx_list : list[int]
            List of lexicographical indices of the Monomials.

        Raises
        ------
        PolynomialError
            If the number of coefficients and multi-indices are not equal.

        See Also
        --------
        MultiIndex
        """
        if len(coef_list) != len(idx_list):
            raise PolynomialError(
                "number of coefficients and multi-indices must be equal"
            )
        for i, c in enumerate(coef_list):
            self.add_monomial_with_idx(c, idx_list[i])
        self.consolidate()

    def is_zero(self) -> bool:
        """
        Is True if the Polynomial is zero.

        Returns
        -------
        bool
            True if the Polynomial is zero.
        """
        self.consolidate()
        return len(self.monos) == 0

    def set_to_zero(self) -> None:
        """
        Set the Polynomial to zero.

        Notes
        -----
        This is done by removing all Monomials from the Polynomial.
        """
        self.monos = []

    @deprecated(
        version="0.5.0",
        reason="Call the object directly (see __call__ method)",
    )
    def _eval(self, x: FloatLike, y: FloatLike) -> FloatLike:
        """
        Evaluate the Polynomial at the point (x, y).

        Parameters
        ----------
        x : FloatLike
            x-coordinate of the point.
        y : FloatLike
            y-coordinate of the point.

        Returns
        -------
        np.ndarray
            Value of the Polynomial at the point (x, y).
        """
        return self(x, y)

    def pow(self, exponent: int) -> Polynomial:
        """
        Raise the Polynomial to a nonnegative integer power.

        Parameters
        ----------
        exponent : int
            Nonnegative integer exponent.

        Returns
        -------
        Polynomial
            Polynomial raised to the exponent.
        """
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Exponent must be nonnegative integer")
        new = Polynomial([(1.0, 0, 0)])
        for _ in range(exponent):
            new *= self
        return new

    def compose(self, q1: Polynomial, q2: Polynomial) -> Polynomial:
        """
        Compose the Polynomial with two other Polynomials.

        Parameters
        ----------
        q1 : Polynomial
            Polynomial in x and y.
        q2 : Polynomial
            Polynomial in x and y.

        Returns
        -------
        Polynomial
            Composed Polynomial new(x,y) = self(q1(x,y), q2(x,y)).
        """
        new = Polynomial()
        for m in self.monos:
            temp = q1.pow(m.alpha.x)
            temp *= q2.pow(m.alpha.y)
            new += m.coef * temp
        return new

    def partial_deriv(self, var: str) -> Polynomial:
        """
        Partial derivative of the Polynomial with respect to the variable var.

        Parameters
        ----------
        var : str
            Variable with respect to which the derivative is taken. Must be
            either 'x' or 'y'.

        Returns
        -------
        Polynomial
            Partial derivative of the Polynomial with respect to the variable
            var.
        """
        new = Polynomial()
        for m in self.monos:
            dm = m.partial_deriv(var)
            new.add_monomial(dm)
        return new

    def grad(self) -> tuple[Polynomial, Polynomial]:
        """
        Gradient of the Polynomial.

        Returns
        -------
        tuple[Polynomial, Polynomial]
            Pair of the partial derivatives of the Polynomial with respect to
            x and y.
        """
        gx = self.partial_deriv("x")
        gy = self.partial_deriv("y")
        return gx, gy

    def laplacian(self) -> Polynomial:
        """
        Laplacian of the Polynomial.

        Returns
        -------
        Polynomial
            Laplacian of the Polynomial: Delta f = f_{xx} + f_{yy}.
        """
        gx, gy = self.grad()
        gxx = gx.partial_deriv("x")
        gyy = gy.partial_deriv("y")
        return gxx + gyy

    def anti_laplacian(self) -> Polynomial:
        """
        Polynomial anti-Laplacian of the Polynomial.

        Returns
        -------
        Polynomial
            Polynomial P such that Delta P = self.
        """
        new = Polynomial()

        # define |(x, y)|^2 = x^2 + y^2
        p1 = Polynomial()
        p1.add_monomials_with_idxs([1, 1], [3, 5])

        # loop over Monomial terms
        for m in self.monos:
            # anti-Laplacian of the Monomial m
            N = m.alpha.order // 2

            # (x ^ 2 + y ^ 2) ^ {k + 1}
            pk = p1.copy()

            # Delta ^ k (x ^ 2 + y ^ 2) ^ alpha
            Lk = Polynomial()
            Lk.add_monomial(m)

            # first term: k = 0
            scale = 0.25 / (1 + m.alpha.order)
            P_alpha = pk * Lk * scale

            # sum over k = 1 : N
            for k in range(1, N + 1):
                pk *= p1
                Lk = Lk.laplacian()
                scale *= -0.25 / ((k + 1) * (m.alpha.order + 1 - k))
                P_alpha += pk * Lk * scale

            # add c_alpha * P_alpha to new
            new += P_alpha

        return new

    def get_weighted_normal_derivative(self, K: MeshCell) -> np.ndarray:
        """
        Compute the weighted normal derivative of the Polynomial.

        Parameters
        ----------
        K : MeshCell
            MeshCell on whose boundary to compute the normal derivative.

        Returns
        -------
        np.ndarray
            Values of the weighted normal derivative of the Polynomial.
        """
        x1, x2 = K.get_boundary_points()
        gx, gy = self.grad()
        gx_trace = gx(x1, x2)
        gy_trace = gy(x1, x2)
        nd = K.dot_with_normal(gx_trace, gy_trace)  # type: ignore
        return K.multiply_by_dx_norm(nd)

    def __repr__(self) -> str:
        """
        Get a string representation of the Polynomial.

        Returns
        -------
        str
            String representation of the Polynomial.
        """
        self.sort()
        if len(self.monos) == 0:
            return "+ (0) "
        msg = ""
        for m in self.monos:
            msg += m.__repr__()
        return msg

    def __str__(self) -> str:
        """
        Get a string representation of the Polynomial.

        Returns
        -------
        str
            String representation of the Polynomial.
        """
        return self.__repr__()

    def __call__(self, x: FloatLike, y: FloatLike) -> FloatLike:
        """
        Evaluate the Polynomial at the point (x, y).

        Parameters
        ----------
        x : FloatLike
            x-coordinate of the point.
        y : FloatLike
            y-coordinate of the point.

        Returns
        -------
        FloatLike
            Value(s) of the Polynomial at the point(s) (x, y). If x and y are
            arrays, returns an array of values of the same shape.
        """
        val = np.zeros(np.shape(x))
        for m in self.monos:
            val += m.eval(x, y)
        return val

    def __eq__(self, other: object) -> bool:
        """
        Test equality between self and other.

        Parameters
        ----------
        other : object
            Object to compare with self. Must be a Polynomial.
        """
        if not isinstance(other, Polynomial):
            raise TypeError("Cannot compare Polynomial to non-Polynomial")
        if len(self.monos) != len(other.monos):
            return False
        self.sort()
        other.sort()
        for i, m in enumerate(self.monos):
            if m != other.monos[i]:
                return False
        return True

    def __add__(self, other: object) -> Polynomial:
        """
        Define the addition operation self + other.

        Parameters
        ----------
        other : object
            Object to add to self. Must be either a Polynomial or a scalar.
        """
        if isinstance(other, Polynomial):
            new = Polynomial()
            for m in self.monos:
                new.add_monomial(m)
            for m in other.monos:
                new.add_monomial(m)
        elif isinstance(other, (int, float)):
            new = Polynomial()
            for m in self.monos:
                new.add_monomial(m)
            constant = Monomial()
            constant.set_multidx_from_idx(0)
            constant.set_coef(other)
            new.add_monomial(constant)
        else:
            raise TypeError(
                "Addition with a Polynomial must be with a scalar or"
                + " with another Polynomial"
            )
        new.consolidate()
        return new

    def __radd__(self, other: object) -> Polynomial:
        """
        Define the addition operator other + self.

        Parameters
        ----------
        other : object
            Object to add to self. Must be either a Polynomial or a scalar.
        """
        if isinstance(other, (int, float)):
            return self + other
        raise TypeError(
            "Addition with a Polynomial must be with a scalar or"
            + " with another Polynomial"
        )

    def __iadd__(self, other: object) -> Polynomial:
        """
        Define the increment operation self += other.

        Parameters
        ----------
        other : object
            Object to add to self. Must be either a Polynomial or a scalar.
        """
        if isinstance(other, Polynomial):
            for m in other.monos:
                self.add_monomial(m)
            self.consolidate()
        elif isinstance(other, (int, float)):
            constant = Monomial()
            constant.set_multidx_from_idx(0)
            constant.set_coef(other)
            self.add_monomial(constant)
            self.consolidate()
        else:
            raise TypeError(
                "Can only add Polynomials to other Polynomials" + " or scalars"
            )
        return self

    def __mul__(self, other: object) -> Polynomial:
        """
        Define the multiplication operator self * other.

        Parameters
        ----------
        other : object
            Object to multiply with self. Must be either a Polynomial or a
            scalar.
        """
        if isinstance(other, Polynomial):
            new = Polynomial()
            for m in self.monos:
                for n in other.monos:
                    new.add_monomial(m * n)
            new.consolidate()
            return new
        if isinstance(other, (int, float)):
            new = Polynomial()
            for m in self.monos:
                new.add_monomial(other * m.copy())
            return new
        raise TypeError(
            "Multiplication by Polynomial must be by a scalar or"
            + " by another Polynomial"
        )

    def __rmul__(self, other: object) -> Polynomial:
        """
        Define the multiplication operator other * self.

        Parameters
        ----------
        other : object
            Object to multiply with self. Must be either a Polynomial or a
            scalar.
        """
        if isinstance(other, Polynomial):
            return self * other
        if isinstance(other, (int, float)):
            return self * other
        raise TypeError(
            "Multiplication by Polynomial must be by a scalar or"
            + " by another Polynomial"
        )

    def __truediv__(self, other: object) -> Polynomial:
        """
        Divide the Polynomial by a scalar.

        Parameters
        ----------
        other : object
            Scalar to divide the Polynomial by.
        """
        if isinstance(other, (int, float)):
            return self * (1 / other)
        raise TypeError("Division of a Polynomial must be by a scalar")

    def __neg__(self) -> Polynomial:
        """Negate the Polynomial."""
        return -1 * self

    def __sub__(self, other: object) -> Polynomial:
        """
        Define the subtraction operator self - other.

        Parameters
        ----------
        other : object
            Object to subtract from self. Must be either a Polynomial or a
            scalar.
        """
        if isinstance(other, (int, float, Polynomial)):
            return self + (-1 * other)
        raise TypeError("Subtraction of a Polynomial must be from a Polynomial")

    def __rsub__(self, other: object) -> Polynomial:
        """
        Define the subtraction operator other - self.

        Parameters
        ----------
        other : object
            Object to subtract self from. Must be either a Polynomial or a
            scalar.
        """
        if isinstance(other, (int, float, Polynomial)):
            return other + (-1 * self)
        raise TypeError("Subtraction of a Polynomial must be from a Polynomial")

    def __pow__(self, exponent: int) -> Polynomial:
        """
        Define the exponentiation operator self ** exponent.

        Parameters
        ----------
        exponent : int
            Nonnegative integer exponent.
        """
        return self.pow(exponent)
