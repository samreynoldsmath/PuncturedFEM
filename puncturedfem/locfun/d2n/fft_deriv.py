"""
fft_deriv.py
============

Fourier (anti-)differentiation of a periodic continuously differentiable
function f: [0,L] -> R sampled on a uniform grid of N points.

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
    Returns df / dx sampled on a uniform grid of N points via the FFT. Assumes
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
    Returns f sampled on a uniform grid of N points via the FFT, up to an
    additive constant. Assumes that df is periodic with period interval_length
    and that df is continuous.
    """
    N = len(df)
    omega = np.fft.fft(df)
    fft_idx = np.fft.fftfreq(len(df))
    fft_idx[0] = 1
    omega *= -1j / (N * fft_idx)
    omega *= 0.5 * interval_length / np.pi
    omega[0] = 0
    return np.real(np.fft.ifft(omega))
