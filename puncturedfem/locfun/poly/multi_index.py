"""
Multi-indices for constructing polynomials.

Classes
-------
MultiIndex
"""

from __future__ import annotations

from math import floor, sqrt
from typing import Optional

from .poly_exceptions import MultiIndexError


class MultiIndex:
    """
    Integer multi-index with two components.

    Attributes
    ----------
    x : int
        First component of the multi-index.
    y : int
        Second component of the multi-index.
    order : int
        Order of the multi-index.
    idx : int
        Unique identifier of the multi-index.

    Notes
    -----
    - A lexicographical ordering of the multi-indices can be obtained by
      associating a unique index k to each multi-index alpha. The index k is
      given by k = alpha_2 + ( m + 1 choose 2 ) where m = alpha_1 + alpha_2.
    - The multi-index alpha can be recovered from the index k by setting
      alpha_1 = m - k + ( m + 1 choose 2 ) and alpha_2 = k - ( m + 1 choose 2 ),
      where m = floor( ( -1 + sqrt(1 + 8k) ) / 2 ).
    """

    x: int
    y: int
    order: int
    idx: int

    def __init__(self, alpha: Optional[list[int]] = None) -> None:
        """
        Integer multi-index with two components.

        Parameters
        ----------
        alpha : list[int], optional
            List of two integers representing the multi-index. Default is None.
        """
        if alpha is None:
            alpha = [0, 0]
        self.set(alpha)

    def validate(self, alpha: list[int]) -> None:
        """Validate the multi-index alpha."""
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
        Set the multi-index to alpha.

        Parameters
        ----------
        alpha : list[int]
            List of two integers representing the multi-index.
        """
        self.validate(alpha)
        self.x = alpha[0]
        self.y = alpha[1]
        self.order = alpha[0] + alpha[1]
        self.idx = alpha[1] + self.order * (self.order + 1) // 2

    def set_from_idx(self, idx: int) -> None:
        """
        Set the multi-index from its index.

        idx : int
            Index identified with multi-index via lexicographical ordering.
        """
        t = floor((sqrt(8 * idx + 1) - 1) / 2)
        N = t * (t + 1) // 2
        alpha = []
        alpha.append(t - idx + N)
        alpha.append(idx - N)
        self.set(alpha)

    def copy(self) -> MultiIndex:
        """
        Return a copy of self.

        Returns
        -------
        MultiIndex
            A copy of self.
        """
        return MultiIndex([self.x, self.y])

    def __eq__(self, other: object) -> bool:
        """
        Return True iff self and other have the same multi-index.

        Parameters
        ----------
        other : object
            Object to compare with self. Must be a MultiIndex.

        Returns
        -------
        bool
            True iff self and other are the same multi-index.
        """
        if not isinstance(other, MultiIndex):
            raise TypeError(
                "Comparison of multi-index to object" + " of different type"
            )
        return self.idx == other.idx

    def __add__(self, other: object) -> MultiIndex:
        """
        Define the operation self + other.

        Parameters
        ----------
        other : object
            Object to add to self. Must be a MultiIndex.
        """
        if isinstance(other, MultiIndex):
            beta = [self.x + other.x, self.y + other.y]
            return MultiIndex(beta)
        raise TypeError("Cannot add multi-index to different type")
