"""
Teardrop shape
x(t) = (2 sin(t/2), −β sin t), β = tan(π/(2α)), α = 3/2
"""

import numpy as np

def _x(t, **kwargs):
	alpha = 3/2
	beta = np.tan(0.5 * np.pi / alpha)
	x = np.zeros((2,len(t)))
	x[0,:] = 2 * np.sin(t / 2)
	x[1,:] = -beta * np.sin(t)
	return x

def _dx(t, **kwargs):
	alpha = 3/2
	beta = np.tan(0.5 * np.pi / alpha)
	dx = np.zeros((2,len(t)))
	dx[0,:] = np.cos(t / 2)
	dx[1,:] = -beta * np.cos(t)
	return dx

def _ddx(t, **kwargs):
	alpha = 3/2
	beta = np.tan(0.5 * np.pi / alpha)
	ddx = np.zeros((2,len(t)))
	ddx[0,:] = -0.5 * np.sin(t / 2)
	ddx[1,:] = beta * np.sin(t)
	return ddx