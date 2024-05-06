"""
Differentiation using FFT.

A periodic continuously differentiable function f: [0,L] -> R sampled on a
uniform grid of N points can be differentiated using the FFT. The derivative
df/dt is computed by taking the FFT of f, multiplying by 1j * N * omega, and
taking the inverse FFT. The antiderivative of df/dt can be computed by taking
the FFT of df/dt, multiplying by -1j / (N * omega), and taking the inverse FFT.

Routines in this module
-----------------------
fft_derivative(f, interval_length)
fft_antiderivative(df, interval_length)

Notes
-----
Uses the fast Fourier transform (FFT) from numpy.fft.
"""

import numpy as np


def fft_derivative(f: np.ndarray, interval_length: float) -> np.ndarray:
    """
    First derivative of a periodic function.

    Parameters
    ----------
    f : np.ndarray
        Values of the function to differentiate.
    interval_length : float
        Period of the function.

    Returns
    -------
    df : np.ndarray
        Derivative of f.

    Notes
    -----
    Returns df/dt sampled on a uniform grid of N points via the FFT. Assumes
    that f is periodic with period interval_length and that f is continuously
    differentiable.
    """
    N = len(f)
    omega = np.fft.fft(f)
    omega *= 1j * N * np.fft.fftfreq(N)
    omega *= 2 * np.pi / interval_length
    return np.real(np.fft.ifft(omega))


def fft_antiderivative(df: np.ndarray, interval_length: float) -> np.ndarray:
    """
    Antiderivative of a periodic function.

    Parameters
    ----------
    df : np.ndarray
        Values of the derivative to integrate.
    interval_length : float
        Period of the function.

    Returns
    -------
    f : np.ndarray
        Antiderivative of df.

    Notes
    -----
    Returns the antiderivative of df/dt sampled on a uniform grid of N points
    via the FFT. Assumes that f is periodic with period interval_length and
    that df is continuous.
    """
    N = len(df)
    omega = np.fft.fft(df)
    fft_idx = np.fft.fftfreq(len(df))
    fft_idx[0] = 1
    omega *= -1j / (N * fft_idx)
    omega *= 0.5 * interval_length / np.pi
    omega[0] = 0
    return np.real(np.fft.ifft(omega))
