"""
Exceptions for the Polynomial and related classes.

Exceptions:
    - DegenerateTriangleError
    - DegreeError
    - InvalidVariableError
    - MultiIndexError
    - PolynomialError
"""


class DegenerateTriangleError(Exception):
    """
    Exception raised if triangle is degenerate.

    A triangle is degenerate if its area is zero.
    """

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class DegreeError(Exception):
    """
    Exception raised if Polynomial degree is not valid.

    The degree of a Polynomial is the maximum degree of its terms, and must be a non-negative integer.
    """

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class InvalidVariableError(Exception):
    """
    Exception raised if variable is not valid.

    A variable is valid if it is a string, either 'x' or 'y'.
    """

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class MultiIndexError(Exception):
    """
    Exception raised if multi-index is not valid.

    A multi-index is valid if it is a pair of non-negative integers.
    """

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg


class PolynomialError(Exception):
    """
    Exception raised if Polynomial is not valid.

    A Polynomial is valid if it is a list of terms, where each term is a pair of a coefficient and a multi-index.
    """

    def __init__(self, msg: str = "") -> None:
        super().__init__()
        self.message = msg
