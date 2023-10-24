"""
Vert.py
=======

Module for the Vert class, which represents a vertex in a planar mesh.
"""


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

    def __eq__(self, other: object) -> bool:
        """
        Returns True if two Vertices have the same coordinates.
        """
        if not isinstance(other, Vert):
            raise TypeError("Cannot compare Vert to non-Vert")
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        """
        Returns a hash of the vertex.
        """
        return hash((self.x, self.y))