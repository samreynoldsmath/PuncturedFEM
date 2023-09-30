"""
vert.py
=======

Module for the vert class, which represents a vertex in a planar mesh.
"""


class vert:
    """
    Represents a vertex in a planar mesh. Stores the vertex's global id and
    coordinates.
    """

    id: int
    x: float
    y: float

    def __init__(self, x: float, y: float, id: int = -1) -> None:
        """
        Constructor for the vert class.

        Parameters
        ----------
        x : float
            The x-coordinate of the vertex.
        y : float
            The y-coordinate of the vertex.
        """
        self.set_coord(x, y)
        self.set_id(id)

    def set_id(self, id: int) -> None:
        """
        Sets the global vertex index.
        """
        if not isinstance(id, int):
            raise TypeError("id must be an integer")
        self.id = id

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
        Returns True if two vertices have the same coordinates.
        """
        if not isinstance(other, vert):
            raise TypeError("Cannot compare vert to non-vert")
        return self.x == other.x and self.y == other.y
