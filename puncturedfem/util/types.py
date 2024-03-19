"""
types.py
========

This module contains type aliases used throughout the puncturedfem package.
"""

import inspect
from typing import Callable, Union

import numpy as np

FloatLike = Union[int, float, np.ndarray]
Func_R2_R = Callable[[FloatLike, FloatLike], FloatLike]


def is_Func_R2_R(func: Func_R2_R) -> bool:
    """
    Check if a function is a map from R^2 to R.

    Parameters
    ----------
    func : Func_R2_R
        The function to be checked.

    Returns
    -------
    bool
        True if the function is a map from R^2 to R, False otherwise.
    """
    if not callable(func):
        return False
    sig = inspect.signature(func)
    params = sig.parameters.values()
    # TODO: have a more flexible check
    return len(params) == 2  # and all(
    # isinstance(param.annotation, type)
    # and issubclass(param.annotation, (int, float, np.ndarray))
    # for param in params
    # )
