"""
test_fft_deriv.py
=================

Test the FFT derivative and antiderivative functions.
"""

from dataclasses import dataclass

import numpy as np

from puncturedfem.locfun.fft_deriv import fft_antiderivative, fft_derivative

TOL = 1e-6


@dataclass
class Interval:
    """
    Interval class for testing FFT derivative.
    """

    def __init__(self, a: float, b: float, n: int = 16):
        self.a = a
        self.b = b
        self.L = b - a
        self.n = n
        self.h = self.L / (2 * self.n)
        self.t = np.linspace(self.a, self.b - self.h, 2 * self.n)


def set_up_interval() -> Interval:
    """
    Set up the test.
    """
    n = 32
    a = np.pi + 1.23
    b = 7 * np.pi + 1.23
    ab = Interval(a, b, n)
    return ab


def get_derivative_error(x: np.ndarray, dx: np.ndarray, L: float) -> float:
    """
    Get the error between the computed derivative and the exact derivative.
    """
    dx_computed = fft_derivative(x, L)
    dx_error = np.abs(dx - dx_computed)
    max_dx_error = np.max(dx_error)
    return max_dx_error


def get_antiderivative_error(dx: np.ndarray, x: np.ndarray, L: float) -> float:
    """
    Get the error between the computed antiderivative and an exact
    antiderivative.
    """
    x_computed = fft_antiderivative(dx, L)
    x_computed += x[0] - x_computed[0]
    x_error = np.abs(x - x_computed)
    max_x_error = np.max(x_error)
    return max_x_error


def constant_function(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a constant function and its derivative.
    """
    x = np.ones(np.shape(t))
    dx = np.zeros(np.shape(t))
    return x, dx


def cos_function(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a cosine function and its derivative.
    """
    x = np.cos(t)
    dx = -np.sin(t)
    return x, dx


def linear_combo_function(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a 5*cos(3t) - (1/7)*sin(2t) function and its derivative.
    """
    x = 5 * np.cos(3 * t) - (1 / 7) * np.sin(2 * t)
    dx = -15 * np.sin(3 * t) - (2 / 7) * np.cos(2 * t)
    return x, dx


def polynomial_function(
    t: np.ndarray, a: float, b: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a 16*(t-a)^2*(t-b)^2/(b-a)^4 function and its derivative.
    """
    L = b - a
    p = 4
    coef = (2 / L) ** (2 * p)
    x = coef * ((t - a) * (t - b)) ** p
    dx = coef * p * (2 * t - a - b) * ((t - a) * (t - b)) ** (p - 1)
    return x, dx


def test_fft_deriv_constant() -> None:
    """
    Test FFT derivative on a constant function.
    """
    ab = set_up_interval()
    x, dx = constant_function(ab.t)
    max_dx_error = get_derivative_error(x, dx, ab.L)
    assert max_dx_error < TOL


def test_fft_deriv_cos() -> None:
    """
    Test FFT derivative on a cosine function.
    """
    ab = set_up_interval()
    x, dx = cos_function(ab.t)
    max_dx_error = get_derivative_error(x, dx, ab.L)
    assert max_dx_error < TOL


def test_fft_deriv_linear_combo() -> None:
    """
    Test FFT derivative on a 5*cos(3t) - (1/7)*sin(2t) function.
    """
    ab = set_up_interval()
    x, dx = linear_combo_function(ab.t)
    max_dx_error = get_derivative_error(x, dx, ab.L)
    assert max_dx_error < TOL


def test_fft_deriv_polynomial() -> None:
    """
    Test FFT derivative on a 16*(t-a)^2*(t-b)^2/(b-a)^4 function.
    """
    ab = set_up_interval()
    x, dx = polynomial_function(ab.t, ab.a, ab.b)
    max_dx_error = get_derivative_error(x, dx, ab.L)
    assert max_dx_error < TOL


def test_fft_antideriv_constant() -> None:
    """
    Test FFT antiderivative on a constant function.
    """
    ab = set_up_interval()
    x, dx = constant_function(ab.t)
    max_x_error = get_antiderivative_error(dx, x, ab.L)
    assert max_x_error < TOL


def test_fft_antideriv_cos() -> None:
    """
    Test FFT antiderivative on a cosine function.
    """
    ab = set_up_interval()
    x, dx = cos_function(ab.t)
    max_x_error = get_antiderivative_error(dx, x, ab.L)
    assert max_x_error < TOL


def test_fft_antideriv_linear_combo() -> None:
    """
    Test FFT antiderivative on a 5*cos(3t) - (1/7)*sin(2t) function.
    """
    ab = set_up_interval()
    x, dx = linear_combo_function(ab.t)
    max_x_error = get_antiderivative_error(dx, x, ab.L)
    assert max_x_error < TOL


def test_fft_antideriv_polynomial() -> None:
    """
    Test FFT antiderivative on a 16*(t-a)^2*(t-b)^2/(b-a)^4 function.
    """
    ab = set_up_interval()
    x, dx = polynomial_function(ab.t, ab.a, ab.b)
    max_x_error = get_antiderivative_error(dx, x, ab.L)
    assert max_x_error < TOL
