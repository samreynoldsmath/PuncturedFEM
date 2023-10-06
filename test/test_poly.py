"""
	Run tests with
	python3 -m unittest
"""
import os
import sys
import unittest

sys.path.append(os.path.abspath("../puncturedfem"))

import numpy as np

from puncturedfem import Polynomial


class TestTemplate(unittest.TestCase):
    def setUp(self):
        """
        z[x,y] = 0
        """
        self.z = Polynomial()

        """
		p[x_, y_] :=
			1
			- 5 x
			+ 2 y
			+ x^2
			- y^2
		"""
        self.p = Polynomial(
            [[1.0, 0, 0], [-5.0, 1, 0], [2.0, 0, 1], [1.0, 2, 0], [-1.0, 0, 2]],
        )

        self.scalar = 4
        self.p_plus_scalar = Polynomial(
            [
                [1.0 + self.scalar, 0, 0],
                [-5.0, 1, 0],
                [2.0, 0, 1],
                [1.0, 2, 0],
                [-1.0, 0, 2],
            ]
        )
        self.p_times_scalar = Polynomial(
            [
                [1.0 * self.scalar, 0, 0],
                [-5.0 * self.scalar, 1, 0],
                [2.0 * self.scalar, 0, 1],
                [1.0 * self.scalar, 2, 0],
                [-1.0 * self.scalar, 0, 2],
            ]
        )

        """
		q[x_, y_] :=
			3
			- 2 x
			+ y
			+ 5 x*y^2
		"""
        self.q = Polynomial(
            [[3.0, 0, 0], [-2.0, 1, 0], [1.0, 0, 1], [5.0, 1, 2]],
        )

        """
		p * q =
			3
			- 17 x
			+ 13 x^2
			- 2 x^3
			+ 7 y
			- 9 x y
			+ x^2 y
			- y^2
			+ 7 x y^2
			- 25 x^2 y^2
			+ 5 x^3 y^2
			- y^3
			+ 10 x y^3
			- 5 x y^4
		"""
        self.pq = Polynomial(
            [
                [3.0, 0, 0],
                [-17.0, 1, 0],
                [13.0, 2, 0],
                [-2.0, 3, 0],
                [7.0, 0, 1],
                [-9.0, 1, 1],
                [1.0, 2, 1],
                [-1.0, 0, 2],
                [7.0, 1, 2],
                [-25.0, 2, 2],
                [5.0, 3, 2],
                [-1.0, 0, 3],
                [10.0, 1, 3],
                [-5.0, 1, 4],
            ]
        )

    def tearDown(self):
        pass

    def test_equality(self):
        self.assertTrue(self.z == self.z)
        self.assertTrue(self.p == self.p)
        self.assertTrue(self.q == self.q)
        self.assertFalse(self.p == self.z)
        self.assertFalse(self.p == self.q)

    def test_inequality(self):
        self.assertFalse(self.z != self.z)
        self.assertFalse(self.p != self.p)
        self.assertFalse(self.q != self.q)
        self.assertTrue(self.p != self.z)
        self.assertTrue(self.p != self.q)

    def test_addition_with_zero(self):
        self.assertTrue(self.z + self.z == self.z)
        self.assertTrue(self.p + self.z == self.p)
        self.assertTrue(self.z + self.p == self.p)

    def test_addition_commutativity(self):
        self.assertTrue(self.p + self.q == self.q + self.p)

    def test_addition_with_scalar(self):
        self.assertTrue(self.p + 0 == self.p)
        self.assertTrue(self.p + self.scalar == self.p_plus_scalar)
        self.assertTrue(self.scalar + self.p == self.p_plus_scalar)

    def test_addition_increment(self):
        r = Polynomial()
        r += self.p
        self.assertTrue(r == self.p)

    def test_addition_increment_with_scalar(self):
        r = Polynomial() + self.p
        r += self.scalar
        self.assertTrue(r == self.p_plus_scalar)
        r -= 4
        r -= self.p
        r += self.q
        self.assertTrue(r == self.q)

    def test_multiplication(self):
        self.assertTrue(self.z * self.p == self.z)
        self.assertTrue(self.p * self.q == self.pq)

    def test_multiplication_with_scalar(self):
        self.assertTrue(0 * self.p == self.z)
        self.assertTrue(self.p * 0 == self.z)
        self.assertTrue(self.scalar * self.p == self.p_times_scalar)
        self.assertTrue(self.p * self.scalar == self.p_times_scalar)

    def test_multiplication_increment(self):
        r = Polynomial() + self.q
        r *= self.p
        self.assertTrue(r == self.pq)
        r *= self.z
        self.assertTrue(r == self.z)

    def test_multiplication_increment_scalar(self):
        r = Polynomial() + self.p
        r *= self.scalar
        self.assertTrue(r == self.p_times_scalar)

        r *= 0
        self.assertTrue(r == self.z)

    def test_gradient(self):
        """
        grad p = {-5 + 2 x, 2 - 2 y}
        grad_q = {-2 + 5 y^2, 1 + 10 x y}
        """

        px = Polynomial([[-5.0, 0, 0], [2.0, 1, 0]])

        py = Polynomial([[2.0, 0, 0], [-2.0, 0, 1]])

        qx = Polynomial([[-2.0, 0, 0], [5.0, 0, 2]])

        qy = Polynomial([[1.0, 0, 0], [10.0, 1, 1]])

        PX, PY = self.p.grad()
        QX, QY = self.q.grad()

        self.assertTrue(PX == px)
        self.assertTrue(PY == py)
        self.assertTrue(QX == qx)
        self.assertTrue(QY == qy)

    def test_laplacian(self):
        """
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

        dq = Polynomial(
            [
                [10.0, 1, 0],
            ]
        )

        dpq = Polynomial(
            [
                [24.0, 0, 0],
                [2.0, 1, 0],
                [-50.0, 2, 0],
                [10.0, 3, 0],
                [-4.0, 0, 1],
                [60.0, 1, 1],
                [-50.0, 0, 2],
                [-30.0, 1, 2],
            ]
        )

        self.assertTrue(self.p.laplacian() == self.z)
        self.assertTrue(self.q.laplacian() == dq)
        self.assertTrue(self.pq.laplacian() == dpq)

    def test_anti_laplacian(self):
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

        self.assertTrue(self.p.anti_laplacian() == P)
        self.assertTrue(P.laplacian() == self.p)

    def test_evaluation(self):
        t = np.linspace(0, 2 * np.pi)
        x = np.cos(t)
        y = np.sin(t)
        val1 = self.p.eval(x, y)
        val2 = 1 - 5 * x + 2 * y + x * x - y * y
        self.assertAlmostEqual(0, np.linalg.norm(val1 - val2))


if __name__ == "__main__":
    unittest.main()
