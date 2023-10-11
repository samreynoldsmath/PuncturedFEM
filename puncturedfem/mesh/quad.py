"""
Quad.py
=======

Module for the Quad class, which represents a 1-dimensional Quadrature object.
"""

import numpy as np


class Quad:
    """
    Quad: 1-dimensional Quadrature object

    Attributes:
        type: str = label for Quadrature variant
        n: int = interval sampled at 2*n points, excluding the last endpoint
        h: float = pi / n sample spacing in tau
        t: array (len=2*n) of sampled parameter between 0 and 2*pi
        wgt: array (len=2*n) of values of lambda'(tau)

    Comment:
        Defaults to the trapezoid rule with n = 16.
        Kress parameter defaults to p = 7.
    """

    type: str
    n: int
    h: float
    t: np.ndarray
    wgt: np.ndarray

    def __init__(self, qtype: str = "trap", n: int = 16, p: int = 7) -> None:
        """
        Constructor for Quad object.

        Parameters
        ----------
        qtype : str, optional
            Label for Quadrature variant. Default is "trap".
        n : int, optional
            Interval sampled at 2*n points, excluding the last endpoint.
            Default is 16.
        p : int, optional
            Kress parameter. Default is 7.
        """
        self.type = qtype
        self.n = n
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
        """ "
        Print method
        """
        msg = f"Quad object \n\ttype\t{self.type} \n\tn\t{self.n}"
        return msg

    def trap(self) -> None:
        """
        Trapezoid rule (default)

        Technically, this defines a left-hand sum. But in our context,
        all functions are periodic, since we are parameterizing closed
        contours.
        """
        self.wgt = np.ones((2 * self.n + 1,))

    def kress(self, p: int) -> None:
        """
        Kress Quadrature

        Used to parameterize an Edge that terminates at corners.

        For a complete description, see:

        R. Kress, A Nyström method for boundary integral equations in domains
        with corners, Numer. Math., 58 (1990), pp. 145-161.
        """

        if p < 2:
            raise ValueError("Kress parameter p must be an integer at least 2")

        # self.type += f'_{p}'

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
        Martensen Quadrature

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