"""
Parameterization of an ellipse centered at the origin
"""

from typing import Any

import numpy as np


def X(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization"""
    x = np.zeros((2, len(t)))
    x[0, :] = kwargs["a"] * np.cos(t)
    x[1, :] = kwargs["b"] * np.sin(t)
    return x


def DX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization derivative"""
    dx = np.zeros((2, len(t)))
    dx[0, :] = -kwargs["a"] * np.sin(t)
    dx[1, :] = kwargs["b"] * np.cos(t)
    return dx


def DDX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization second derivative"""
    ddx = np.zeros((2, len(t)))
    ddx[0, :] = -kwargs["a"] * np.cos(t)
    ddx[1, :] = -kwargs["b"] * np.sin(t)
    return ddx
