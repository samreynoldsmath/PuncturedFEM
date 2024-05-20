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

from ..mesh.cell import MeshCell


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


def get_weighted_tangential_derivative_from_trace(
    K: MeshCell, f_vals: np.ndarray
) -> np.ndarray:
    """
    Compute the weighted tangential derivative from the trace.

    Parameters
    ----------
    K : MeshCell
        Mesh cell.
    f_vals : np.ndarray
        Values of the function on the boundary.

    Returns
    -------
    wtd : np.ndarray
        Weighted tangential derivative.
    """
    wtd = np.zeros((K.num_pts,))

    for i in range(K.num_holes + 1):
        # get indices of this contour
        j = K.component_start_idx[i]
        jp1 = K.component_start_idx[i + 1]

        # get values on this contour
        f_vals_c = f_vals[j:jp1]

        # compute weighted tangential derivative on this contour
        interval_length = 2 * np.pi * K.components[i].num_edges
        dfc_dt_wgt = fft_derivative(f_vals_c, interval_length)

        # add to weighted tangential derivative on the whole boundary
        wtd[j:jp1] = dfc_dt_wgt

    return wtd


def fft_antiderivative_on_each_component(
    K: MeshCell, f_prime: np.ndarray
) -> np.ndarray:
    """
    Compute the antiderivative on each component.

    Parameters
    ----------
    K : MeshCell
        The mesh cell over whose boundary we are integrating.
    f_prime : np.ndarray
        The derivative of the function to integrate. The length of f_prime
        should be equal to the number of points on the boundary, and sampled
        values should represent the trace of a continuous function.

    Returns
    -------
    f : np.ndarray
        The antiderivative of the function on each component.
    """
    f = np.zeros((K.num_pts,))
    for j in range(K.num_holes + 1):
        pt_idx = slice(K.component_start_idx[j], K.component_start_idx[j + 1])
        interval_length = 2 * np.pi * K.components[j].num_edges
        f[pt_idx] = fft_antiderivative(f_prime[pt_idx], interval_length)
    return f
