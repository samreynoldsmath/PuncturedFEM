"""
Parameterization of the unit circle centered at the origin
"""

import numpy as np

def _x(t, **kwargs):
    if 'radius' in kwargs:
        R = kwargs['radius']
    else:
        R = 1
    x = np.zeros((2, len(t)))
    x[0,:] = R * np.cos(t)
    x[1,:] = R * np.sin(t)
    return x

def _dx(t, **kwargs):
    if 'radius' in kwargs:
        R = kwargs['radius']
    else:
        R = 1
    dx = np.zeros((2, len(t)))
    dx[0,:] = -R * np.sin(t)
    dx[1,:] = R * np.cos(t)
    return dx

def _ddx(t, **kwargs):
    if 'radius' in kwargs:
        R = kwargs['radius']
    else:
        R = 1
    ddx = np.zeros((2, len(t)))
    ddx[0,:] = -R * np.cos(t)
    ddx[1,:] = -R * np.sin(t)
    return ddx