"""
Parameterization of a circular arc joining the origin to (1,0).
The center of the circle lies at (1/2, H). The points on the circle below
the x2 axis are discarded.
"""

from typing import Any

import numpy as np


def unpack(kwargs: Any) -> tuple[float, float, float, float]:
    """
    Extract the parameters H, R, t0, omega from the keyword arguments
    """
    H = kwargs["H"]
    R = np.sqrt(0.25 + H**2)
    t0 = np.arcsin(H / R)
    omega = -0.5 - t0 / np.pi
    return H, R, t0, omega


def X(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization"""
    H, R, t0, omega = unpack(kwargs)
    theta = t0 + np.pi + omega * t
    x = np.zeros((2, len(t)))
    x[0, :] = 0.5 + R * np.cos(theta)
    x[1, :] = H + R * np.sin(theta)
    return x


def DX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization derivative"""
    H, R, t0, omega = unpack(kwargs)
    theta = t0 + np.pi + omega * t
    dx = np.zeros((2, len(t)))
    dx[0, :] = -omega * R * np.sin(theta)
    dx[1, :] = +omega * R * np.cos(theta)
    return dx


def DDX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization second derivative"""
    H, R, t0, omega = unpack(kwargs)
    theta = t0 + np.pi + omega * t
    ddx = np.zeros((2, len(t)))
    ddx[0, :] = -omega * omega * R * np.cos(theta)
    ddx[1, :] = -omega * omega * R * np.sin(theta)
    return ddx
