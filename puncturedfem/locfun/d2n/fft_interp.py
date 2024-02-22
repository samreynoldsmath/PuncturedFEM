"""
fft_interp.py
=============

Trigonometric interpolation using FFT.
"""

import numpy as np


def fft_interpolation(f: np.ndarray, interp: int) -> np.ndarray:
    """
    Given f sampled on a uniform grid of N points, such that f periodic with
    period interval_length and f continuous, returns f sampled interpolated to
    N * interp equispaced points on the same interval. Uses FFT.
    """

    if not isinstance(interp, int):
        raise TypeError("interp must be an integer >= 1")
    if interp < 1:
        raise ValueError("interp must be an integer >= 1")
    if interp == 1:
        return f

    N = len(f)
    M = N * interp

    omega = np.fft.fft(f)
    omega = np.fft.fftshift(omega)
    omega_interp = np.zeros((M,), dtype=complex)
    omega_interp[(M -N)//2:(M-N)//2+N] = omega
    omega_interp = np.fft.ifftshift(omega_interp)
    F = np.real(np.fft.ifft(omega_interp)) * interp

    return F
