"""
Parameterization of the unit circle centered at the origin
"""

from typing import Any

import numpy as np


def X(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization"""
    if "radius" in kwargs:
        R = kwargs["radius"]
    else:
        R = 1
    x = np.zeros((2, len(t)))
    x[0, :] = R * np.cos(t)
    x[1, :] = R * np.sin(t)
    return x


def DX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization derivative"""
    if "radius" in kwargs:
        R = kwargs["radius"]
    else:
        R = 1
    dx = np.zeros((2, len(t)))
    dx[0, :] = -R * np.sin(t)
    dx[1, :] = R * np.cos(t)
    return dx


def DDX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization second derivative"""
    if "radius" in kwargs:
        R = kwargs["radius"]
    else:
        R = 1
    ddx = np.zeros((2, len(t)))
    ddx[0, :] = -R * np.cos(t)
    ddx[1, :] = -R * np.sin(t)
    return ddx
