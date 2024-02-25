"""
Vert.py
=======

Module for the Vert class, which represents a vertex in a planar mesh.
"""

from __future__ import annotations


class Vert:
    """
    Represents a vertex in a planar mesh. Stores the vertex's global index and
    coordinates.

    Attributes
    ----------
    idx : int
        The global index of the vertex.
    x : float
        The x-coordinate of the vertex.
    y : float
        The y-coordinate of the vertex.
    """

    idx: int
    x: float
    y: float

    def __init__(self, x: float, y: float, idx: int = -1) -> None:
        """
        Constructor for the Vert class.

        Parameters
        ----------
        x : float
            The x-coordinate of the vertex.
        y : float
            The y-coordinate of the vertex.
        """
        self.set_coord(x, y)
        self.set_idx(idx)

    def set_idx(self, idx: int) -> None:
        """
        Sets the global vertex index.
        """
        if not isinstance(idx, int):
            raise TypeError("idx must be an integer")
        self.idx = idx

    def set_coord(self, x: float, y: float) -> None:
        """
        Sets the coordinates of the vertex.
        """
        if isinstance(x, int):
            x = float(x)
        if isinstance(y, int):
            y = float(y)
        if not isinstance(x, float) or not isinstance(y, float):
            raise TypeError("Coordinates x and y must be floats")
        self.x = x
        self.y = y

    def __str__(self) -> str:
        """
        Returns a string representation of the vertex.
        """
        return f"[vert{self.idx}]({self.x}, {self.y})"

    def __repr__(self) -> str:
        """
        Returns a string representation of the vertex.
        """
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        """
        Returns True if two Vertices have the same coordinates.
        """
        if not isinstance(other, Vert):
            return NotImplemented
        diff = self - other
        if isinstance(diff, Vert):
            return diff.norm() < 1e-12
        raise TypeError("Subtraction against Vert must return a Vert")

    def __add__(self, other: object) -> Vert:
        """
        Returns a new Vert with coordinates equal to the sum of the
        coordinates of the two Verts.
        """
        if not isinstance(other, Vert):
            raise TypeError("Cannot add non-Vert to Vert")
        return Vert(self.x + other.x, self.y + other.y)

    def __sub__(self, other: object) -> Vert:
        """
        Returns a new Vert with coordinates equal to the difference of the
        coordinates of the two Verts.
        """
        if not isinstance(other, Vert):
            raise TypeError("Cannot subtract non-Vert from Vert")
        return Vert(self.x - other.x, self.y - other.y)

    def __mul__(self, other: object) -> Vert:
        """
        Returns a new Vert with coordinates equal to the product of the
        coordinates of the two Verts, or the coordinates of the Vert multiplied
        by the scalar.
        """
        if isinstance(other, (int, float)):
            return Vert(self.x * other, self.y * other)
        if isinstance(other, Vert):
            return Vert(self.x * other.x, self.y * other.y)
        raise TypeError(
            "Multiplication against Vert must be by int, float, or Vert"
        )

    def __rmul__(self, other: object) -> Vert:
        """
        Returns a new Vert with coordinates equal to the product of the
        coordinates of the two Verts, or the coordinates of the Vert multiplied
        by the scalar.
        """
        if isinstance(other, (int, float)):
            return Vert(self.x * other, self.y * other)
        if isinstance(other, Vert):
            return Vert(self.x * other.x, self.y * other.y)
        raise TypeError(
            "Multiplication against Vert must be by int, float, or Vert"
        )

    def __truediv__(self, other: object) -> Vert:
        """
        Returns a new Vert with coordinates equal to the quotient of the
        coordinates of the two Verts, or the coordinates of the Vert divided by
        the scalar.
        """
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return Vert(self.x / other, self.y / other)
        if isinstance(other, Vert):
            if other.x == 0 or other.y == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            return Vert(self.x / other.x, self.y / other.y)
        raise TypeError("Division against Vert must be by int, float, or Vert")

    def norm(self) -> float:
        """
        Returns the Euclidean norm of the vertex.
        """
        return (self.x**2 + self.y**2) ** 0.5
