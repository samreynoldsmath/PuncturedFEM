"""
Parameterization of a line joining the origin to (1,0).
"""

import numpy as np

def _x(t, **kwargs):
    x = np.zeros((2,len(t)))
    x[0,:] = t / (2*np.pi)
    return x

def _dx(t, **kwargs):
    dx = np.zeros((2,len(t)))
    dx[0,:] = np.ones((len(t),)) / (2*np.pi)
    return dx

def _ddx(t, **kwargs):
    ddx = np.zeros((2,len(t)))
    return ddx