"""
Parameterization of the unit circle centered at the origin
"""

import numpy as np

def _x(t, **kwargs):
    x = np.zeros((2,len(t)))
    x[0,:] = np.cos(t)
    x[1,:] = np.sin(t)
    return x

def _dx(t, **kwargs):
    dx = np.zeros((2,len(t)))
    dx[0,:] = - np.sin(t)
    dx[1,:] = np.cos(t)
    return dx

def _ddx(t, **kwargs):
    ddx = np.zeros((2,len(t)))
    ddx[0,:] = - np.cos(t)
    ddx[1,:] = - np.sin(t)
    return ddx