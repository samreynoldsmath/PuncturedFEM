"""
Module for printing colored text to the terminal.

Classes
-------
Color
    Enum for terminal colors.

Functions
---------
print_color
    Print a string in color.
"""

from enum import Enum


class Color(Enum):
    """
    Enum for terminal colors.

    Attributes
    ----------
    RESET : str
        Reset the terminal color.
    RED : str
        Red terminal color.
    GREEN : str
        Green terminal color.
    YELLOW : str
        Yellow terminal color.
    BLUE : str
        Blue terminal color.
    MAGENTA : str
        Magenta terminal color.
    CYAN : str
        Cyan terminal color.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    def __str__(self) -> str:
        """Return the name of the color."""
        return self.name


def print_color(s: str, c: Color) -> None:
    """
    Print a string in color.

    Parameters
    ----------
    s : str
        The string to be printed.
    c : Color
        The color in which to print the string.
    """
    print(c.value + s + Color.RESET.value)
