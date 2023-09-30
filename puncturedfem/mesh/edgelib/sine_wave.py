"""
Parameterization of sine wave joining the origin to (1,0) of the form

    x = t / (2*pi)
    y = a * sin( omega/2 * t ) , 0 < t < 2*pi

Include arguments for the amplitude ('amp') and the frequency ('freq').
The frequency argument must be an integer.
"""

from typing import Any

import numpy as np


def unpack(kwargs: Any) -> tuple[float, float]:
    """
    Extract the parameters a, omega from the keyword arguments
    """
    if "amp" not in kwargs:
        raise ValueError("amp must be specified")
    if "freq" not in kwargs:
        raise ValueError("freq must be specified")
    a = kwargs["amp"]
    omega = kwargs["freq"]
    return a, omega


def X(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization"""
    a, omega = unpack(kwargs)
    x = np.zeros((2, len(t)))
    x[0, :] = t / (2 * np.pi)
    x[1, :] = a * np.sin(omega * t / 2)
    return x


def DX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization derivative"""
    a, omega = unpack(kwargs)
    dx = np.zeros((2, len(t)))
    dx[0, :] = np.ones((len(t),)) / (2 * np.pi)
    dx[1, :] = 0.5 * a * omega * np.cos(omega * t / 2)
    return dx


def DDX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization second derivative"""
    a, omega = unpack(kwargs)
    ddx = np.zeros((2, len(t)))
    ddx[1, :] = -0.25 * a * omega * omega * np.sin(omega * t / 2)
    return ddx
