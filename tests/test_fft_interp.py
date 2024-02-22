"""
test_fft_interp.py
=================

Test the FFT interpolation function.
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from puncturedfem.locfun.d2n.fft_interp import fft_interpolation

TOL = 1e-6

SAVE_PLOT = True


@dataclass
class Interval:
    """
    Interval class for testing FFT derivative.
    """

    def __init__(self, a: float, b: float, interp: int, n: int = 16):
        self.a = a
        self.b = b
        self.L = b - a
        self.n = n
        self.h = self.L / (2 * self.n)
        self.t = np.linspace(self.a, self.b - self.h, 2 * self.n)
        self.interp = interp
        self.N = n * interp
        self.H = self.L / (2 * self.N)
        self.T = np.linspace(self.a, self.b - self.H, 2 * self.N)


def set_up_interval() -> Interval:
    """
    Set up the test.
    """
    n = 32
    interp = 4
    a = np.pi + 1.23
    b = 7 * np.pi + 1.23
    ab = Interval(a, b, interp, n)
    return ab


def get_interpolation_error(
    x: np.ndarray, X: np.ndarray, ab: Interval, fig_name: str
) -> float:
    """
    Get the error between the computed interpolation and the exact
    interpolation.
    """
    X_computed = fft_interpolation(x, ab.interp)
    X_error = np.abs(X - X_computed)

    if SAVE_PLOT:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(ab.T, X_computed, "ro")
        plt.plot(ab.T, X, "k.")
        plt.subplot(2, 1, 2)
        if np.linalg.norm(X_error) > 1e-12:
            plt.semilogy(ab.T, X_error, "ro")
        else:
            plt.plot(ab.T, X_error, "ro")
        plt.savefig("tests/figs/" + fig_name)
        plt.close()

    max_X_error = np.max(X_error)
    return max_X_error


def constant_function(t: np.ndarray) -> np.ndarray:
    """
    Return a constant function and its derivative.
    """
    return np.ones(np.shape(t))


def cos_function(t: np.ndarray) -> np.ndarray:
    """
    Return a cosine function and its derivative.
    """
    return np.cos(t)


def linear_combo_function(t: np.ndarray) -> np.ndarray:
    """
    Return a 5*cos(3t) - (1/7)*sin(2t) function and its derivative.
    """
    return 5 * np.cos(3 * t) - (1 / 7) * np.sin(2 * t)


def polynomial_function(t: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Return a 16*(t-a)^2*(t-b)^2/(b-a)^4 function and its derivative.
    """
    L = b - a
    p = 4
    coef = (2 / L) ** (2 * p)
    return coef * ((t - a) * (t - b)) ** p


def test_fft_interp_constant() -> None:
    """
    Test FFT interpolation on a constant function.
    """
    ab = set_up_interval()
    x = constant_function(ab.t)
    X = constant_function(ab.T)
    max_dx_error = get_interpolation_error(x, X, ab, "constant.png")
    assert max_dx_error < TOL


def test_fft_interp_cos() -> None:
    """
    Test FFT derivative on a cosine function.
    """
    ab = set_up_interval()
    x = cos_function(ab.t)
    X = cos_function(ab.T)
    max_dx_error = get_interpolation_error(x, X, ab, "cos.png")
    assert max_dx_error < TOL


def test_fft_interp_linear_combo() -> None:
    """
    Test FFT derivative on a 5*cos(3t) - (1/7)*sin(2t) function.
    """
    ab = set_up_interval()
    x = linear_combo_function(ab.t)
    X = linear_combo_function(ab.T)
    max_dx_error = get_interpolation_error(x, X, ab, "linear_combo.png")
    assert max_dx_error < TOL


def test_fft_interp_polynomial() -> None:
    """
    Test FFT derivative on a 16*(t-a)^2*(t-b)^2/(b-a)^4 function.
    """
    ab = set_up_interval()
    x = polynomial_function(ab.t, ab.a, ab.b)
    X = polynomial_function(ab.T, ab.a, ab.b)
    max_dx_error = get_interpolation_error(x, X, ab, "polynomial.png")
    assert max_dx_error < TOL
