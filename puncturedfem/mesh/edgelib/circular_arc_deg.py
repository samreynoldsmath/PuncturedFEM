"""
Parameterization of a circular arc from (1,0) to (cos(theta0), sin(theta0))
"""

import numpy as np

def unpack(kwargs):
	theta0 = kwargs['theta0']
	if theta0 <= 0 or theta0 > 360:
		raise ValueError('theta0 must be a nontrivial angle between '
		   + '0 and 360 degrees')
	omega = theta0 / 360.0
	return omega

def _x(t, **kwargs):
	omega = unpack(kwargs)
	x = np.zeros((2,len(t)))
	x[0,:] = np.cos(omega * t)
	x[1,:] = np.sin(omega * t)
	return x

def _dx(t, **kwargs):
	omega = unpack(kwargs)
	dx = np.zeros((2,len(t)))
	dx[0,:] = -omega * np.sin(omega * t)
	dx[1,:] = +omega * np.cos(omega * t)
	return dx

def _ddx(t, **kwargs):
	omega = unpack(kwargs)
	ddx = np.zeros((2,len(t)))
	ddx[0,:] = -omega * omega * np.cos(omega * t)
	ddx[1,:] = -omega * omega * np.sin(omega * t)
	return ddx