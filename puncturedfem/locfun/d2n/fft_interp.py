"""
fft_interp.py
=============

Trigonometric interpolation using FFT.
"""

import numpy as np

from ...mesh.cell import MeshCell


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
    omega_interp[(M - N) // 2 : (M - N) // 2 + N] = omega
    omega_interp = np.fft.ifftshift(omega_interp)
    F = np.real(np.fft.ifft(omega_interp)) * interp

    return F


def interpolate_on_boundary(
    f_vals: np.ndarray, K: MeshCell, K_interp: MeshCell
) -> np.ndarray:
    """
    Interpolates a trace on the boundary.
    """

    if len(f_vals) != K_interp.num_pts:
        raise ValueError("f_vals must be same length as K_interp.num_pts")

    F_vals = np.zeros((K.num_pts,))

    # loop over boundary components
    for i in range(K.num_holes + 1):
        # get indices of this component on reduced sampling
        j_interp = K_interp.component_start_idx[i]
        jp1_interp = K_interp.component_start_idx[i + 1]

        # get values on this contour
        f_vals_c = f_vals[j_interp:jp1_interp]

        # get interpolated values
        F_vals_c = fft_interpolation(f_vals_c, K.interp)

        # get indices of this component on standard sampling
        j = K.component_start_idx[i]
        jp1 = K.component_start_idx[i + 1]

        # set values
        F_vals[j:jp1] = F_vals_c

    return F_vals
