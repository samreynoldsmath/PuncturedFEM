"""
test_poly.py
============

Tests the Polynomial class.
"""

from dataclasses import dataclass

import numpy as np

from puncturedfem import Polynomial


@dataclass
class PolynomialCollection:
    """
    Set up some polynomials for testing.
    """

    def __init__(self) -> None:
        """
        Set up some polynomials for testing.
        """

        self.z = Polynomial()  # zero polynomial

        self.p = Polynomial(
            [
                (1.0, 0, 0),  # 1
                (-5.0, 1, 0),  # -5 x
                (2.0, 0, 1),  # 2 y
                (1.0, 2, 0),  # x^2
                (-1.0, 0, 2),  # -y^2
            ]
        )

        self.scalar = 4
        self.p_plus_scalar = Polynomial(
            [
                (1.0 + self.scalar, 0, 0),  # 1 + scalar
                (-5.0, 1, 0),  # -5 x
                (2.0, 0, 1),  # 2 y
                (1.0, 2, 0),  # x^2
                (-1.0, 0, 2),  # -y^2
            ]
        )
        self.p_times_scalar = Polynomial(
            [
                (1.0 * self.scalar, 0, 0),  # scalar
                (-5.0 * self.scalar, 1, 0),  # -5 scalar x
                (2.0 * self.scalar, 0, 1),  # 2 scalar y
                (1.0 * self.scalar, 2, 0),  # scalar x^2
                (-1.0 * self.scalar, 0, 2),  # -scalar y^2
            ]
        )

        self.q = Polynomial(
            [
                (3.0, 0, 0),  # 3
                (-2.0, 1, 0),  # -2 x
                (1.0, 0, 1),  # y
                (5.0, 1, 2),  # 5 x y^2
            ]
        )

        self.pq = Polynomial(
            [
                (3.0, 0, 0),  # 3
                (-17.0, 1, 0),  # -17 x
                (13.0, 2, 0),  # 13 x^2
                (-2.0, 3, 0),  # -2 x^3
                (7.0, 0, 1),  # 7 y
                (-9.0, 1, 1),  # -9 x y
                (1.0, 2, 1),  # x^2 y
                (-1.0, 0, 2),  # -y^2
                (7.0, 1, 2),  # 7 x y^2
                (-25.0, 2, 2),  # -25 x^2 y^2
                (5.0, 3, 2),  # 5 x^3 y^2
                (-1.0, 0, 3),  # -y^3
                (10.0, 1, 3),  # 10 x y^3
                (-5.0, 1, 4),  # -5 x y^4
            ]
        )


def test_equality() -> None:
    """
    Test the == operator.
    """
    pc = PolynomialCollection()
    assert pc.z == pc.z
    assert pc.p == pc.p
    assert pc.q == pc.q
    assert not pc.p == pc.z  # pylint: disable=unneeded-not
    assert not pc.p == pc.q  # pylint: disable=unneeded-not


def test_inequality() -> None:
    """
    Test the != operator.
    """
    pc = PolynomialCollection()
    assert not pc.z != pc.z  # pylint: disable=unneeded-not
    assert not pc.p != pc.p  # pylint: disable=unneeded-not
    assert not pc.q != pc.q  # pylint: disable=unneeded-not
    assert pc.p != pc.z
    assert pc.p != pc.q


def test_addition_with_zero() -> None:
    """
    Test addition with the zero polynomial.
    """
    pc = PolynomialCollection()
    assert pc.z + pc.z == pc.z
    assert pc.p + pc.z == pc.p
    assert pc.z + pc.p == pc.p


def test_addition_commutativity() -> None:
    """
    Test addition commutativity.
    """
    pc = PolynomialCollection()
    assert pc.p + pc.q == pc.q + pc.p


def test_addition_with_scalar() -> None:
    """
    Test addition with scalars.
    """
    pc = PolynomialCollection()
    assert pc.p + 0 == pc.p
    assert pc.p + pc.scalar == pc.p_plus_scalar
    assert pc.scalar + pc.p == pc.p_plus_scalar


def test_addition_increment() -> None:
    """
    Test addition increment operator +=
    """
    pc = PolynomialCollection()
    r = Polynomial()
    r += pc.p
    assert r == pc.p


def test_addition_increment_with_scalar() -> None:
    """
    Test addition increment operator += with scalar
    """
    pc = PolynomialCollection()
    r = Polynomial() + pc.p
    r += pc.scalar
    assert r == pc.p_plus_scalar
    r -= 4
    r -= pc.p
    r += pc.q
    assert r == pc.q


def test_multiplication() -> None:
    """
    Test multiplication of polynomials.
    """
    pc = PolynomialCollection()
    assert pc.z * pc.p == pc.z
    assert pc.p * pc.q == pc.pq


def test_multiplication_with_scalar() -> None:
    """
    Test multiplication of polynomials with scalars.
    """
    pc = PolynomialCollection()
    assert 0 * pc.p == pc.z
    assert pc.p * 0 == pc.z
    assert pc.scalar * pc.p == pc.p_times_scalar
    assert pc.p * pc.scalar == pc.p_times_scalar


def test_multiplication_increment() -> None:
    """
    Test multiplication increment operator *=
    """
    pc = PolynomialCollection()
    r = Polynomial() + pc.q
    r *= pc.p
    assert r == pc.pq
    r *= pc.z
    assert r == pc.z


def test_multiplication_increment_scalar() -> None:
    """
    Test multiplication increment operator *= with scalar
    """
    pc = PolynomialCollection()
    r = Polynomial() + pc.p
    r *= pc.scalar
    assert r == pc.p_times_scalar

    r *= 0
    assert r == pc.z


def test_gradient() -> None:
    """
    Test gradient of polynomials.

    grad p = {-5 + 2 x, 2 - 2 y}
    grad_q = {-2 + 5 y^2, 1 + 10 x y}
    """
    pc = PolynomialCollection()

    px = Polynomial([(-5.0, 0, 0), (2.0, 1, 0)])

    py = Polynomial([(2.0, 0, 0), (-2.0, 0, 1)])

    qx = Polynomial([(-2.0, 0, 0), (5.0, 0, 2)])

    qy = Polynomial([(1.0, 0, 0), (10.0, 1, 1)])

    PX, PY = pc.p.grad()
    QX, QY = pc.q.grad()

    assert PX == px
    assert PY == py
    assert QX == qx
    assert QY == qy


def test_laplacian() -> None:
    """
    Test Laplacian of polynomials.

    Delta p = 0
    Delta q = 10 x
    Delta (pq) =
            24
            + 2 x
            - 50 x^2
            + 10 x^3
            - 4 y
            + 60 x y
            - 50 y^2
            - 30 x y^2
    """
    pc = PolynomialCollection()

    dq = Polynomial(
        [
            (10.0, 1, 0),
        ]
    )

    dpq = Polynomial(
        [
            (24.0, 0, 0),
            (2.0, 1, 0),
            (-50.0, 2, 0),
            (10.0, 3, 0),
            (-4.0, 0, 1),
            (60.0, 1, 1),
            (-50.0, 0, 2),
            (-30.0, 1, 2),
        ]
    )

    assert pc.p.laplacian() == pc.z
    assert pc.q.laplacian() == dq
    assert pc.pq.laplacian() == dpq


def test_anti_laplacian() -> None:
    """
    Test anti-Laplacian of polynomials.
    """
    pc = PolynomialCollection()

    P = Polynomial()
    P.add_monomials_with_idxs(
        coef_list=[
            1 / 4,
            1 / 4,
            -5 / 8,
            -5 / 8,
            1 / 4,
            1 / 4,
            7 / 96,
            6 / 96,
            -1 / 96,
            1 / 96,
            -6 / 96,
            -7 / 96,
        ],
        idx_list=[3, 5, 6, 8, 7, 9, 10, 12, 14, 10, 12, 14],
    )

    assert pc.p.anti_laplacian() == P
    assert P.laplacian() == pc.p


def test_evaluation() -> None:
    """
    Test evaluation of polynomials.
    """
    pc = PolynomialCollection()
    t = np.linspace(0, 2 * np.pi)
    x = np.cos(t)
    y = np.sin(t)
    val1 = pc.p.eval(x, y)
    val2 = 1 - 5 * x + 2 * y + x * x - y * y
    assert np.linalg.norm(val1 - val2) < 1e-12
