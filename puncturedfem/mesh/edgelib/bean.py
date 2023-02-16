"""
Bean shape
x(t) = (cos t + 0.65 cos(2t), 1.5 sin t)
"""

import numpy as np

def _x(t, **kwargs):
	x = np.zeros((2,len(t)))
	x[0,:] = np.cos(t) + 0.65 * np.cos(2 * t)
	x[1,:] = 1.5 * np.sin(t)
	return x

def _dx(t, **kwargs):
	dx = np.zeros((2,len(t)))
	dx[0,:] = - np.sin(t) - 2 * 0.65 * np.sin(2 * t)
	dx[1,:] = 1.5 * np.cos(t)
	return dx

def _ddx(t, **kwargs):
	ddx = np.zeros((2,len(t)))
	ddx[0,:] = - np.cos(t) - 4 * 0.65 * np.cos(2 * t)
	ddx[1,:] = - 1.5 * np.sin(t)
	return ddx