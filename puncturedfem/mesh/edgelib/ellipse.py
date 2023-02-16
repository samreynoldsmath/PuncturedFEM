"""
Parameterization of an ellipse centered at the origin
"""

import numpy as np

def _x(t, **kwargs):
    x = np.zeros((2,len(t)))
    x[0,:] = kwargs['a'] * np.cos(t)
    x[1,:] = kwargs['b'] * np.sin(t)
    return x

def _dx(t, **kwargs):
    dx = np.zeros((2,len(t)))
    dx[0,:] = - kwargs['a'] * np.sin(t)
    dx[1,:] = kwargs['b'] * np.cos(t)
    return dx

def _ddx(t, **kwargs):
    ddx = np.zeros((2,len(t)))
    ddx[0,:] = - kwargs['a'] * np.cos(t)
    ddx[1,:] = - kwargs['b'] * np.sin(t)
    return ddx