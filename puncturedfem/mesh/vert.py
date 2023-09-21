class vert:
    """Stores physical location of a mesh vertex"""

    id: int
    x: float
    y: float

    def __init__(self, x: float, y: float, id: int = -1) -> None:
        self.set_coord(x, y)
        self.set_id(id)

    def set_id(self, id: int) -> None:
        if not isinstance(id, int):
            raise TypeError("id must be an integer")
        self.id = id

    def set_coord(self, x: float, y: float) -> None:
        if isinstance(x, int):
            x = float(x)
        if isinstance(y, int):
            y = float(y)
        if not isinstance(x, float) or not isinstance(y, float):
            raise TypeError("Coordinates x and y must be floats")
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, vert):
            raise TypeError("Cannot compare vert to non-vert")
        return self.x == other.x and self.y == other.y
