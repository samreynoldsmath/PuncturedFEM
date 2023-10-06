"""
multi_index.py
==============

Module containing the MultiIndex class, which is used to represent
multi-indices of the form
    alpha = (alpha_1, alpha_2)
where alpha_1 and alpha_2 are nonnegative integers.
"""

from __future__ import annotations

from math import floor, sqrt
from typing import Optional

from .poly_exceptions import MultiIndexError


class MultiIndex:
    """
    Integer multi-index with two components
    """

    x: int
    y: int
    order: int
    idx: int

    def __init__(self, alpha: Optional[list[int]] = None) -> None:
        """
        Constructor for MultiIndex class.

        Parameters
        ----------
        alpha : list[int], optional
            List of two integers representing the multi-index. Default is None.
        """
        if alpha is None:
            alpha = [0, 0]
        self.set(alpha)

    def validate(self, alpha: list[int]) -> None:
        """
        Validates the multi-index alpha.
        """
        if not isinstance(alpha, list):
            raise TypeError("Multi-index must be list of two integers")
        if len(alpha) != 2:
            raise MultiIndexError("Multi-index is assumed to have 2 components")
        if not (isinstance(alpha[0], int) and isinstance(alpha[1], int)):
            raise TypeError("Multi-index must be list of two integers")
        if alpha[0] < 0 or alpha[1] < 0:
            raise ValueError("Components of multi-index must be nonnegative")

    def set(self, alpha: list[int]) -> None:
        """
        Sets the multi-index to alpha.
        """
        self.validate(alpha)
        self.x = alpha[0]
        self.y = alpha[1]
        self.order = alpha[0] + alpha[1]
        self.idx = alpha[1] + self.order * (self.order + 1) // 2

    def set_from_idx(self, idx: int) -> None:
        """
        Sets the multi-index from its id.
        """
        t = floor((sqrt(8 * idx + 1) - 1) / 2)
        N = t * (t + 1) // 2
        alpha = []
        alpha.append(t - idx + N)
        alpha.append(idx - N)
        self.set(alpha)

    def copy(self) -> MultiIndex:
        """
        Returns a copy of self.
        """
        return MultiIndex([self.x, self.y])

    def __eq__(self, other: object) -> bool:
        """
        Returns True iff self and other have the same multi-index.
        """
        if not isinstance(other, MultiIndex):
            raise TypeError(
                "Comparison of multi-index to object" + " of different type"
            )
        return self.idx == other.idx

    def __add__(self, other: object) -> MultiIndex:
        """
        Defines the operation self + other
        where other is a multi-index
        """
        if isinstance(other, MultiIndex):
            beta = [self.x + other.x, self.y + other.y]
            return MultiIndex(beta)
        raise TypeError("Cannot add multi-index to different type")
