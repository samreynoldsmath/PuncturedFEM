"""
mesh_exceptions.py
==================

Module containing exceptions for the mesh subpackage.
"""


class EdgeTransformationError(Exception):
    """Exception raised if Edge transformation is not valid"""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class EmbeddingError(Exception):
    """Exception raised if embedding is not valid"""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class InteriorPointError(Exception):
    """Exception raised if point is not in interior of mesh"""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class NotParameterizedError(Exception):
    """Exception raised if edges are not parameterized"""

    def __init__(self, cant_do: str = "calling this method") -> None:
        super().__init__()
        self.message = "Must parameterize edges before " + cant_do


class SizeMismatchError(Exception):
    """Exception raised if array sizes do not match"""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg
