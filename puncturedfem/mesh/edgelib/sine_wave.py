"""
Parameterization of sine wave joining the origin to (1,0) of the form

    x = t / (2*pi)
    y = a * sin( omega/2 * t ) , 0 < t < 2*pi

Include arguments for the amplitude ('amp') and the frequency ('freq').
The frequency argument must be an integer.
"""

import numpy as np

def _x(t, **kwargs):

    a = kwargs['amp']
    omega = kwargs['freq']

    if np.abs(omega - int(omega)) > 1e-12:
        raise Exception('freq of the sine wave must be an integer')

    x = np.zeros((2,len(t)))
    x[0,:] = t / (2*np.pi)
    x[1,:] = a*np.sin(omega*t/2)
    return x

def _dx(t, **kwargs):
    a = kwargs['amp']
    omega = kwargs['freq']
    dx = np.zeros((2,len(t)))
    dx[0,:] = np.ones((len(t),)) / (2*np.pi)
    dx[1,:] = 0.5*a*omega*np.cos(omega*t/2)
    return dx

def _ddx(t, **kwargs):
    a = kwargs['amp']
    omega = kwargs['freq']
    ddx = np.zeros((2,len(t)))
    ddx[1,:] = -0.25*a*omega*omega*np.sin(omega*t/2)
    return ddx