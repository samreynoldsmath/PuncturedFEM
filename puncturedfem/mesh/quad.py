"""
1-dimensional quadrature object.

Also includes a convenience function for generating a dictionary of Quad
objects.

Classes
-------
Quad
    1-dimensional quadrature object
QuadDict
    Dictionary holding quadratures

Functions
---------
get_quad_dict(n=16, p=7)
    Return a dictionary of Quad objects
"""

from __future__ import annotations

from typing import TypedDict
from warnings import warn

import numpy as np


class QuadDict(TypedDict):
    """
    Dictionary holding quadratures.

    Attributes
    ----------
    trap : Quad
        Trapezoid rule
    kress : Quad
        Kress quadrature
    """

    trap: Quad
    kress: Quad


def get_quad_dict(n: int = 16, p: int = 7) -> QuadDict:
    """
    Return a dictionary of Quad objects.

    Parameters
    ----------
    n : int, optional
        Interval sampled at 2*n points, excluding the last endpoint.
        Default is 16.
    p : int, optional
        Kress parameter.
        Default is 7.

    Returns
    -------
    quad_dict : dict
        Dictionary of Quad objects.
    """
    quad_dict: QuadDict = {
        "trap": Quad(qtype="trap", n=n),
        "kress": Quad(qtype="kress", n=n, p=p),
    }
    return quad_dict


class Quad:
    """
    1-dimensional quadrature object.

    Attributes
    ----------
    type : str
        Label for quadrature variant. Default is "trap", for trapezoid rule.
        Other options are "kress" and "martensen".
    n : int
        Interval sampled at 2*n points, excluding the last endpoint.
    N : int
        Number of points in the quadrature.
    h : float
        Step size.
    t : np.ndarray
        Sample points.
    wgt : np.ndarray
        Quadrature weights.
    """

    type: str
    n: int
    N: int
    h: float
    t: np.ndarray
    wgt: np.ndarray

    def __init__(self, qtype: str = "trap", n: int = 16, p: int = 7) -> None:
        """
        Initialize a Quad object.

        Parameters
        ----------
        qtype : str, optional
            Label for Quadrature variant. Default is "trap".
        n : int, optional
            Interval sampled at 2*n points, excluding the last endpoint.
            Default is 16.
        p : int, optional
            Kress parameter.
            Default is 7.
        """
        self.type = qtype
        self._set_n(n)
        self.h = np.pi / n
        self.t = np.linspace(0, 2 * np.pi, 2 * n + 1)
        if self.type == "kress":
            self.kress(p)
            return
        if self.type == "mart" or type == "martensen":
            self.martensen()
            return
        self.trap()

    def __repr__(self) -> str:
        """Return a string representation of the Quad object."""
        msg = f"Quad object \n\ttype\t{self.type} \n\tn\t{self.n}"
        return msg

    def _set_n(self, n: int) -> None:
        if not isinstance(n, int):
            raise TypeError("Quad parameter n must be an integer")
        if n < 4:
            raise ValueError("Quad parameter n must be at least 4")
        if n > 128:
            warn("Quad: n > 128 may cause numerical instability")
        self.n = n

    def trap(self) -> None:
        """
        Trapezoid rule (default).

        Technically, this defines a left-hand sum. But in our context,
        all functions are periodic, since we are parameterizing closed
        contours.
        """
        self.wgt = np.ones((2 * self.n + 1,))

    def kress(self, p: int) -> None:
        """
        Kress quadrature.

        Used to parameterize an Edge that terminates at corners. For a complete
        description, see:
            R. Kress, A Nyström method for boundary integral equations in
            domains with corners, Numer. Math., 58 (1990), pp. 145-161.
        """
        if p < 2:
            raise ValueError("Kress parameter p must be an integer at least 2")

        s = self.t / np.pi - 1
        s2 = s * s
        c = (0.5 - 1 / p) * s * s2 + s / p + 0.5
        cp = c**p
        denom = cp + (1 - c) ** p

        self.t = (2 * np.pi) * cp / denom
        self.wgt = (
            (3 * (p - 2) * s2 + 2) * (c * (1 - c)) ** (p - 1) / denom**2
        )

    def martensen(self) -> None:
        """
        Martensen Quadrature.

        E. Martensen, Über eine Methode zum räumlichen Neumannschen Problem
        mit einer An-wendung für torusartige Berandungen, Acta Math., 109
        (1963), pp. 75-135.
        """
        self.wgt = np.zeros((2 * self.n + 1,))
        for m in range(1, self.n + 1):
            self.wgt += np.cos(m * self.t) / m
        self.wgt *= 0.5 / self.n

        self.t = 2 * np.sin(self.t / 2)
        self.t *= self.t
