"""
Teardrop shape
x(t) = (2 sin(t/2), −β sin t), β = tan(π/(2α)), α = 3/2 (default)
"""

from typing import Any

import numpy as np


def unpack(kwargs: Any) -> Any:
    """Unpack parameters"""
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
    else:
        alpha = 3 / 2
    if alpha <= 1:
        raise ValueError("alpha must > 1")
    beta = np.tan(0.5 * np.pi / alpha)
    return beta


def X(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization"""
    beta = unpack(kwargs)
    x = np.zeros((2, len(t)))
    x[0, :] = 2 * np.sin(t / 2)
    x[1, :] = -beta * np.sin(t)
    return x


def DX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization derivative"""
    beta = unpack(kwargs)
    dx = np.zeros((2, len(t)))
    dx[0, :] = np.cos(t / 2)
    dx[1, :] = -beta * np.cos(t)
    return dx


def DDX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization second derivative"""
    beta = unpack(kwargs)
    ddx = np.zeros((2, len(t)))
    ddx[0, :] = -0.5 * np.sin(t / 2)
    ddx[1, :] = beta * np.sin(t)
    return ddx
