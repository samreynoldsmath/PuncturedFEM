"""
cubic spline interpolation of an array of points

"""

from typing import Any
import scipy.interpolate as interpolate

import numpy as np
import matplotlib.pyplot as plt

#Takes in a sequence of points and produces a (cubic) spline that interpolates them.
#This could be optimized to only perform this computation once, but a constant factor of 3
#is probably not worth it.
def unpack(kwargs: Any) -> tuple[np.ndarray,np.ndarray,int]:
    tck,u= interpolate.splprep(kwargs['pts'],s=0)
    return tck

def X(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization"""
    x = np.zeros((2, len(t)))
    myx = interpolate.splev(1/2/np.pi*t,unpack(kwargs))
    x[0,:] = myx[0]
    x[1,:] = myx[1]
    return x


def DX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization derivative"""
    dx = np.zeros((2, len(t)))
    mydx = interpolate.splev(1/2/np.pi*t,unpack(kwargs),der=1)
    dx[0,:] = mydx[0]
    dx[1,:] = mydx[1]
    return 1/2/np.pi*dx


def DDX(t: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Edge parametrization second derivative"""
    ddx = np.zeros((2, len(t)))
    myddx = interpolate.splev(1/2/np.pi*t,unpack(kwargs),der=2)
    ddx[0,:] = myddx[0]
    ddx[1,:] = myddx[1]
    return 1/4/np.pi/np.pi*ddx
