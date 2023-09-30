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
        n = 64
        q_trap = pf.quad(qtype="trap", n=n)
        q_kress = pf.quad(qtype="kress", n=n)
        self.quad_dict = {"kress": q_kress, "trap": q_trap}
        self.tol = 1e-14

    def test_integration_circle(self):
        """
        Test integration over cell boundary
        """
        K = self.build_circle()
        boundary_length_exact = np.pi / 2.0

        # calculate length of boundary by integrating 1 over boundary
        one = np.ones((K.num_pts,))
        boundary_length_computed = K.integrate_over_boundary(one)

        # compute error
        boundary_length_error = np.abs(
            boundary_length_exact - boundary_length_computed
        )

        # check that computed length is close to exact length
        self.assertTrue(boundary_length_error < self.tol)

    def test_integration_square(self):
        """
        Test integration over cell boundary
        """
        K = self.build_square()
        boundary_length_exact = 4.0

        # calculate length of boundary by integrating 1 over boundary
        one = np.ones((K.num_pts,))
        boundary_length_computed = K.integrate_over_boundary(one)

        # compute error
        boundary_length_error = np.abs(
            boundary_length_exact - boundary_length_computed
        )

        # check that computed length is close to exact length
        self.assertTrue(boundary_length_error < self.tol)

    def test_integration_punctured_square(self):
        """
        Test integration over cell boundary
        """
        K = self.build_punctured_square()
        boundary_length_exact = 4.0 + np.pi / 2.0

        # calculate length of boundary by integrating 1 over boundary
        one = np.ones((K.num_pts,))
        boundary_length_computed = K.integrate_over_boundary(one)

        # compute error
        boundary_length_error = np.abs(
            boundary_length_exact - boundary_length_computed
        )

        # check that computed length is close to exact length
        self.assertTrue(boundary_length_error < self.tol)

    def build_circle(self):
        # define vertices
        verts: list[pf.vert] = []
        verts.append(pf.vert(x=0.5, y=0.5))  # center of circle

        # define edges
        edges: list[pf.edge] = []
        edges.append(
            pf.edge(
                verts[0],
                verts[0],
                pos_cell_idx=0,
                curve_type="circle",
                quad_type="trap",
                radius=0.25,
            )
        )

        # define mesh cell
        K = pf.cell(idx=0, edges=edges)

        # parameterize edges
        K.parameterize(self.quad_dict)

        return K

    def build_square(self):
        # define vertices
        verts: list[pf.vert] = []
        verts.append(pf.vert(x=0.0, y=0.0))
        verts.append(pf.vert(x=1.0, y=0.0))
        verts.append(pf.vert(x=1.0, y=1.0))
        verts.append(pf.vert(x=0.0, y=1.0))

        # define edges
        edges: list[pf.edge] = []
        edges.append(pf.edge(verts[0], verts[1], pos_cell_idx=0))
        edges.append(pf.edge(verts[1], verts[2], pos_cell_idx=0))
        edges.append(pf.edge(verts[2], verts[3], pos_cell_idx=0))
        edges.append(pf.edge(verts[3], verts[0], pos_cell_idx=0))

        # define mesh cell
        K = pf.cell(idx=0, edges=edges)

        # parameterize edges
        K.parameterize(self.quad_dict)

        return K

    def build_punctured_square(self):
        # define vertices
        verts: list[pf.vert] = []
        verts.append(pf.vert(x=0.0, y=0.0))
        verts.append(pf.vert(x=1.0, y=0.0))
        verts.append(pf.vert(x=1.0, y=1.0))
        verts.append(pf.vert(x=0.0, y=1.0))
        verts.append(pf.vert(x=0.5, y=0.5))  # center of circle

        # define edges
        edges: list[pf.edge] = []
        edges.append(pf.edge(verts[0], verts[1], pos_cell_idx=0))
        edges.append(pf.edge(verts[1], verts[2], pos_cell_idx=0))
        edges.append(pf.edge(verts[2], verts[3], pos_cell_idx=0))
        edges.append(pf.edge(verts[3], verts[0], pos_cell_idx=0))
        edges.append(
            pf.edge(
                verts[4],
                verts[4],
                neg_cell_idx=0,
                curve_type="circle",
                quad_type="trap",
                radius=0.25,
            )
        )

        # define mesh cell
        K = pf.cell(idx=0, edges=edges)

        # parameterize edges
        K.parameterize(self.quad_dict)

        return K

    def build_pacman(self):
        pass

    def build_ghost(self):
        pass


if __name__ == "__main__":
    unittest.main()
