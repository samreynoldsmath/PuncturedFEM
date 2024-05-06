"""
Exceptions for the mesh subpackage.

Exceptions:
    EdgeTransformationError: Raised if Edge transformation is not valid.
    EmbeddingError: Raised if embedding is not valid.
    InteriorPointError: Raised if point is not in interior of mesh.
    NotParameterizedError: Raised if edges are not parameterized.
    SizeMismatchError: Raised if array sizes do not match.
"""


class EdgeTransformationError(Exception):
    """Exception raised if Edge transformation is not valid."""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class EmbeddingError(Exception):
    """Exception raised if embedding is not valid."""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class InteriorPointError(Exception):
    """Exception raised if point is not in interior of mesh."""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class NotParameterizedError(Exception):
    """Exception raised if edges are not parameterized."""

    def __init__(self, cant_do: str = "calling this method") -> None:
        super().__init__()
        self.message = "Must parameterize edges before " + cant_do


class SizeMismatchError(Exception):
    """Exception raised if array sizes do not match."""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg
