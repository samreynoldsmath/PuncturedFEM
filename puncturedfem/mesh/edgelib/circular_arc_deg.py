"""
Parameterization of a circular arc from (1,0) to (cos(theta0), sin(theta0))
"""

from typing import Any

import numpy as np


def unpack(kwargs: Any) -> float:
    """
    Extract the angle theta0 from the keyword arguments
    """
    if "theta0" not in kwargs:
        raise ValueError("theta0 must be specified")
    theta0 = kwargs["theta0"]
    if theta0 <= 0 or theta0 > 360:
        raise ValueError(
            "theta0 must be a nontrivial angle between " + "0 and 360 degrees"
        )
    omega = theta0 / 360.0
    return omega


def X(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization"""
    omega = unpack(kwargs)
    x = np.zeros((2, len(t)))
    x[0, :] = np.cos(omega * t)
    x[1, :] = np.sin(omega * t)
    return x


def DX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization derivative"""
    omega = unpack(kwargs)
    dx = np.zeros((2, len(t)))
    dx[0, :] = -omega * np.sin(omega * t)
    dx[1, :] = +omega * np.cos(omega * t)
    return dx


def DDX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization second derivative"""
    omega = unpack(kwargs)
    ddx = np.zeros((2, len(t)))
    ddx[0, :] = -omega * omega * np.cos(omega * t)
    ddx[1, :] = -omega * omega * np.sin(omega * t)
    return ddx
