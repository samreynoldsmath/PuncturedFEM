"""
edge_space.py
=============

Module containing the EdgeSpace class for managing spaces of trace functions.
"""

import numpy as np

from ..mesh.edge import Edge
from ..solver.globkey import GlobalKey
from .poly.barycentric import barycentric_coordinates_edge
from .poly.legendre import integrated_legendre_tensor_products
from .poly.poly import Polynomial


class EdgeSpace:
    """
    The space of Polynomial traces of degree  <= deg on an Edge.
    """

    e: Edge
    deg: int
    vert_fun_traces: list[Polynomial]
    edge_fun_traces: list[Polynomial]
    vert_fun_GlobalKeys: list[GlobalKey]
    edge_fun_GlobalKeys: list[GlobalKey]
    num_vert_funs: int
    num_edge_funs: int
    num_funs: int

    def __init__(self, e: Edge, deg: int) -> None:
        """
        Constructor for EdgeSpace class.

        Parameters
        ----------
        e : Edge
            Edge on which the space is defined.
        deg : int
            Maximum Polynomial degree.
        """
        self.edge_fun_traces = []
        self.vert_fun_traces = []
        self.set_edge(e)
        self.set_deg(deg)
        self.build_spanning_set()
        self.reduce_to_basis()
        self.compute_num_vert_funs()
        self.compute_num_edge_funs()
        self.generate_vert_fun_GlobalKeys()
        self.generate_edge_fun_GlobalKeys()
        self.find_num_funs()

    def set_edge(self, e: Edge) -> None:
        """
        Set the Edge on which the space is defined.
        """
        if not isinstance(e, Edge):
            raise TypeError("e must be an Edge")
        self.e = e

    def set_deg(self, deg: int) -> None:
        """
        Set the maximum Polynomial degree.
        """
        if not isinstance(deg, int):
            raise TypeError("deg must be an integer")
        if deg < 1:
            raise ValueError("deg must be a positive integer")
        self.deg = deg

    def find_num_funs(self) -> None:
        """
        Find the total number of functions in the space.
        """
        self.num_funs = self.num_vert_funs + self.num_edge_funs

    def compute_num_vert_funs(self) -> None:
        """
        Number of vertex functions is set to self.num_vert_funs.
        """
        self.num_vert_funs = len(self.vert_fun_traces)

    def compute_num_edge_funs(self) -> None:
        """
        Number of Edge functions is set to self.num_edge_funs.
        """
        self.num_edge_funs = len(self.edge_fun_traces)

    def generate_vert_fun_GlobalKeys(self) -> None:
        """Generate global keys for Edge and vertex functions"""
        self.vert_fun_GlobalKeys = []
        if self.e.is_loop:
            return
        for k in [self.e.anchor.idx, self.e.endpnt.idx]:
            self.vert_fun_GlobalKeys.append(GlobalKey("Vert", vert_idx=k))

    def generate_edge_fun_GlobalKeys(self) -> None:
        """Generate global keys for Edge and vertex functions"""
        self.edge_fun_GlobalKeys = []
        for k in range(self.num_edge_funs):
            self.edge_fun_GlobalKeys.append(
                GlobalKey("Edge", edge_idx=self.e.idx, EdgeSpace_idx=k)
            )

    def build_spanning_set(self) -> None:
        """
        Spanning set of P_p(e) using traces of
                L_m(x_1) * L_n(x_2) - a * ell_0 - b * ell_1
        where
                L_j is the jth integrated Legendre Polynomial,
                a = L_m(y_1) * L_n(y_2),
                b = L_m(z_1) * L_n(z_2),
                y = (y_1, y_2) starting vertex of Edge
                z = (z_1, z_2) ending vertex of Edge
        """

        # compute Legendre tensor products
        self.edge_fun_traces = integrated_legendre_tensor_products(self.deg)

        linear_fun_idx = [0, 1, self.deg + 1]

        # redo linear functions to avoid constants
        self.edge_fun_traces[linear_fun_idx[0]] = Polynomial(
            [
                # 1 + x + y
                (1.0, 0, 0),
                (1.0, 1, 0),
                (1.0, 0, 1),
            ]
        )
        self.edge_fun_traces[linear_fun_idx[1]] = Polynomial(
            [
                # 1 - x - y
                (+1.0, 0, 0),
                (-1.0, 1, 0),
                (-1.0, 0, 1),
            ]
        )
        self.edge_fun_traces[linear_fun_idx[2]] = Polynomial(
            [
                # x - y
                (+1.0, 1, 0),
                (-1.0, 0, 1),
            ]
        )

        # transform coordinates to bounding box
        self.transform_coordinates_to_bounding_box()

        # different cases for closed loops and edges with distinct endpoints
        if not self.e.is_loop:
            # compute barycentric coordinates
            ell = barycentric_coordinates_edge(self.e)

            # correct Edge functions to vanish at endpoints
            for j in range(2, len(self.edge_fun_traces)):
                # get values of Legendre tensor products at the endpoints
                a0 = self.edge_fun_traces[j].eval(
                    self.e.x[0, 0], self.e.x[1, 0]
                )
                a1 = self.edge_fun_traces[j].eval(
                    self.e.x[0, -1], self.e.x[1, -1]
                )

                # force values at endpoints to be zero
                self.edge_fun_traces[j] -= a0 * ell[0] + a1 * ell[1]

            # set vertex functions to barycentric coordinates
            self.vert_fun_traces.append(ell[0])
            self.vert_fun_traces.append(ell[1])

            # delete other linear functions from Edge function list
            del self.edge_fun_traces[linear_fun_idx[2]]
            del self.edge_fun_traces[linear_fun_idx[1]]
            del self.edge_fun_traces[linear_fun_idx[0]]

            # add linear function vanishing at endpoints to Edge function list
            self.edge_fun_traces.append(ell[2])

    def transform_coordinates_to_bounding_box(self) -> None:
        """
        Transform the domain of the Legendre tensor products from the square
        [-1, 1] x [-1, 1] to the bounding box of the Edge, by composing with
        an affine change of coordinates (i.e. deg=1 Polynomials).
        """

        # get bounding box
        xmin, xmax, ymin, ymax = self.e.get_bounding_box()

        # scaling factors
        sx = (xmax - xmin) / 2
        sy = (ymax - ymin) / 2

        # translation factors
        tx = (xmax + xmin) / 2
        ty = (ymax + ymin) / 2

        # define affine change of coordinates
        qx = Polynomial(
            [
                (-tx / sx, 0, 0),
                (1 / sx, 1, 0),
            ]
        )

        qy = Polynomial(
            [
                (-ty / sy, 0, 0),
                (1 / sy, 0, 1),
            ]
        )

        # map Legendre tensor products from square to bounding box
        for j, f in enumerate(self.edge_fun_traces):
            self.edge_fun_traces[j] = f.compose(qx, qy)

    def reduce_to_basis(self) -> None:
        """
        Reduce spanning set to basis by determining the pivot columns of the
        mass matrix.
        """
        M = self.get_gram_matrix()
        idx = self.get_basis_index_set(M)
        basis = []
        for k in idx:
            basis.append(self.edge_fun_traces[k])
        self.edge_fun_traces = basis

    def get_gram_matrix(self, precond: bool = True) -> np.ndarray:
        """
        Return the mass matrix M_ij = int_e phi_i phi_j ds.
        If precond is True (default), then the normalization preconditioner is
        applied.
        """

        # compute inner products
        m = len(self.edge_fun_traces)
        M = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                integrand = self.edge_fun_traces[i] * self.edge_fun_traces[j]
                M[i, j] = self.e.integrate_over_edge(
                    integrand.eval(x=self.e.x[0, :], y=self.e.x[1, :])
                )
                M[j, i] = M[i, j]

        # eliminate rows and columns with zero diagonals
        M = self.eliminate_zeros(M)

        # apply normalization preconditioner
        if precond:
            M = self.diagonal_rescale(M)

        return M

    def diagonal_rescale(self, M: np.ndarray) -> np.ndarray:
        """
        Return the normalized mass matrix M_ij / sqrt(M_ii * M_jj).
        """
        m = np.shape(M)[0]
        for i in range(m):
            for j in range(i + 1, m):
                M[i, j] /= np.sqrt(M[i, i] * M[j, j])
                M[j, i] = M[i, j]
        for i in range(m):
            M[i, i] = 1
        return M

    def eliminate_zeros(self, M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
        """
        Return the matrix M with rows and columns with zero diagonals removed.
        """
        m = np.shape(M)[0]
        idx = []
        for i in range(m):
            if M[i, i] > tol:
                idx.append(i)
        n = len(idx)
        N = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                N[i, j] = M[idx[i], idx[j]]
                N[j, i] = N[i, j]
        return N

    def get_basis_index_set(
        self, M: np.ndarray, tol: float = 1e-12
    ) -> list[int]:
        """
        Return the index set of the pivot columns of the matrix M.
        """
        idx: list[int] = []
        if np.shape(M)[0] == 0:
            return idx
        k = int(np.argmax(np.diag(M)))
        while M[k, k] > tol:
            idx.append(k)
            M -= np.outer(M[:, k], M[:, k]) / M[k, k]
            k = int(np.argmax(np.diag(M)))
        return sorted(idx)
