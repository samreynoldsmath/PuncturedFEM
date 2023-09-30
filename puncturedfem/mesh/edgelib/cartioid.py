"""
Generalized Teardrop / Cartioid
x(t) = r(t) [cos(t), sin(t)] where
r(t) = 1 + a (1 - t / pi) ^ 8, a > -1 is a fixed parameter
Note:
    a = 0 gives the unit circle
    -1 < a < 0 is "generalized cartioid" with a reentrant corner
    a > 0 is a "generalized teardrop"
"""

from typing import Any

import numpy as np


def X(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization"""
    a = kwargs["a"]
    r = 1 + a * (1 - t / np.pi) ** 8

    x = np.zeros((2, len(t)))
    x[0, :] = r * np.cos(t)
    x[1, :] = r * np.sin(t)
    return x


def DX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization derivative"""
    a = kwargs["a"]
    r = 1 + a * (1 - t / np.pi) ** 8
    dr = -(8 * a / np.pi) * (1 - t / np.pi) ** 7

    dx = np.zeros((2, len(t)))
    dx[0, :] = dr * np.cos(t) - r * np.sin(t)
    dx[1, :] = dr * np.sin(t) + r * np.cos(t)
    return dx


def DDX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization second derivative"""
    a = kwargs["a"]
    r = 1 + a * (1 - t / np.pi) ** 8
    dr = -(8 * a / np.pi) * (1 - t / np.pi) ** 7
    ddr = (56 * a / (np.pi * np.pi)) * (1 - t / np.pi) ** 6

    ddx = np.zeros((2, len(t)))
    ddx[0, :] = (ddr - r) * np.cos(t) - 2 * dr * np.sin(t)
    ddx[1, :] = (ddr - r) * np.sin(t) + 2 * dr * np.cos(t)
    return ddx
