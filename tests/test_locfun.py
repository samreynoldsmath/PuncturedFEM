"""
	Run tests with
	python3 -m unittest
"""
import os
import sys
import unittest

sys.path.append(os.path.abspath("../puncturedfem"))

import numpy as np

import puncturedfem as pf


class TestLocalFunction(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 64
        self.tol = 1e-10

    def test_punctured_square(self):
        """
        Sets up the mesh cell K and functions functions v,w as in
        examples/ex1a-square-hole.ipynb
        """

        K = self.build_punctured_square()
        solver = pf.NystromSolver(K)

        # get the coordinates of sampled boundary points
        x1, x2 = K.get_boundary_points()

        # set target value of logarithmic coefficient
        a_exact = 1

        # set point in hole interior
        xi = [0.5, 0.5]

        # define trace of v
        v_trace = (
            np.exp(x1) * np.cos(x2)
            + 0.5 * a_exact * np.log((x1 - xi[0]) ** 2 + (x2 - xi[1]) ** 2)
            + x1**3 * x2
            + x1 * x2**3
        )

        # create Polynomial object
        v_laplacian = pf.Polynomial([[12.0, 1, 1]])

        # create local function object
        v = pf.LocalFunction(
            nyst=solver, lap_poly=v_laplacian, has_poly_trace=False
        )
        v.set_trace_values(v_trace)
        v.compute_all()

        # trace of w
        w_trace = (
            (x1 - 0.5) / ((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2)
            + x1**3
            + x1 * x2**2
        )

        # define a monomial term by specifying its multi-index and coefficient
        w_laplacian = pf.Polynomial([[8.0, 1, 0]])

        # declare w as local function object
        w = pf.LocalFunction(
            nyst=solver, lap_poly=w_laplacian, has_poly_trace=False
        )
        w.set_trace_values(w_trace)
        w.compute_all()

        # compute L^2 inner product
        l2_vw_exact = 1.39484950156676
        l2_vw_computed = v.get_l2_inner_prod(w)
        l2_error = abs(l2_vw_computed - l2_vw_exact)

        # compare to exact values
        h1_vw_exact = 4.46481780319135
        h1_vw_computed = v.get_h1_semi_inner_prod(w)
        h1_error = abs(h1_vw_computed - h1_vw_exact)

        self.assertTrue(l2_error < self.tol)
        self.assertTrue(h1_error < self.tol)

    def build_punctured_square(self):
        q_trap = pf.Quad(qtype="trap", n=self.n)
        q_kress = pf.Quad(qtype="kress", n=self.n)
        quad_dict = {"kress": q_kress, "trap": q_trap}

        # define vertices
        verts: list[pf.vert] = []
        verts.append(pf.Vert(x=0.0, y=0.0))
        verts.append(pf.Vert(x=1.0, y=0.0))
        verts.append(pf.Vert(x=1.0, y=1.0))
        verts.append(pf.Vert(x=0.0, y=1.0))
        verts.append(pf.Vert(x=0.5, y=0.5))  # center of circle

        # define edges
        edges: list[pf.Edge] = []
        edges.append(pf.Edge(verts[0], verts[1], pos_cell_idx=0))
        edges.append(pf.Edge(verts[1], verts[2], pos_cell_idx=0))
        edges.append(pf.Edge(verts[2], verts[3], pos_cell_idx=0))
        edges.append(pf.Edge(verts[3], verts[0], pos_cell_idx=0))
        edges.append(
            pf.Edge(
                verts[4],
                verts[4],
                neg_cell_idx=0,
                curve_type="circle",
                quad_type="trap",
                radius=0.25,
            )
        )

        # define mesh cell
        K = pf.MeshCell(idx=0, edges=edges)

        # parameterize edges
        K.parameterize(quad_dict)

        return K

    def build_pacman(self):
        pass

    def build_ghost(self):
        pass


if __name__ == "__main__":
    unittest.main()
