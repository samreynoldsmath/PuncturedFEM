"""
Parameterization of a line joining the origin to (1,0).
"""

from typing import Any

import numpy as np


def X(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization"""
    x = np.zeros((2, len(t)))
    x[0, :] = t / (2 * np.pi)
    return x


def DX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization derivative"""
    dx = np.zeros((2, len(t)))
    dx[0, :] = np.ones((len(t),)) / (2 * np.pi)
    return dx


def DDX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization second derivative"""
    ddx = np.zeros((2, len(t)))
    return ddx
