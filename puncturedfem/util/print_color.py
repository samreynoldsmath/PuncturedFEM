"""
print_color.py
==============

Module for printing colored text to the terminal.
"""

from enum import Enum


class Color(Enum):
    """
    Enum for terminal colors.
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    def __str__(self) -> str:
        """
        Return the name of the color.
        """
        return self.name


def print_color(s: str, c: Color) -> None:
    """
    Print a string in color.
    """
    print(c.value + s + Color.RESET.value)
