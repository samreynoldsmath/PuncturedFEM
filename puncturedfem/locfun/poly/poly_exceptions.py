"""
poly_exceptions.py
==================

Module containing exceptions for the poly subpackage.
"""


class DegenerateTriangleError(Exception):
    """Exception raised if triangle is degenerate"""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class DegreeError(Exception):
    """Exception raised if Polynomial degree is not valid"""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class InvalidVariableError(Exception):
    """Exception raised if variable is not valid"""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class MultiIndexError(Exception):
    """Exception raised if multi-index is not valid"""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class PolynomialError(Exception):
    """Exception raised if Polynomial is not valid"""

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg
