import numpy as np

from ..mesh.edge import edge
from ..solver.globkey import global_key
from .poly.barycentric import barycentric_coordinates_edge
from .poly.legendre import integrated_legendre_tensor_products
from .poly.poly import polynomial


class edge_space:
    e: edge
    deg: int
    vert_fun_traces: list[polynomial]
    edge_fun_traces: list[polynomial]
    vert_fun_global_keys: list[global_key]
    edge_fun_global_keys: list[global_key]
    num_vert_funs: int
    num_edge_funs: int
    num_funs: int

    def __init__(self, e: edge, deg: int) -> None:
        self.edge_fun_traces = []
        self.vert_fun_traces = []
        self.set_edge(e)
        self.set_deg(deg)
        self.build_spanning_set()
        self.reduce_to_basis()
        self.compute_num_vert_funs()
        self.compute_num_edge_funs()
        self.generate_vert_fun_global_keys()
        self.generate_edge_fun_global_keys()
        self.find_num_funs()

    def set_edge(self, e: edge) -> None:
        if not isinstance(e, edge):
            raise TypeError("e must be an edge")
        self.e = e

    def set_deg(self, deg: int) -> None:
        if not isinstance(deg, int):
            raise TypeError("deg must be an integer")
        if deg < 1:
            raise ValueError("deg must be a positive integer")
        self.deg = deg

    def find_num_funs(self) -> None:
        self.num_funs = self.num_vert_funs + self.num_edge_funs

    def compute_num_vert_funs(self) -> None:
        self.num_vert_funs = len(self.vert_fun_traces)

    def compute_num_edge_funs(self) -> None:
        self.num_edge_funs = len(self.edge_fun_traces)

    def generate_vert_fun_global_keys(self) -> None:
        """Generate global keys for edge and vertex functions"""
        self.vert_fun_global_keys = []
        if self.e.is_loop:
            return
        for k in [self.e.anchor.id, self.e.endpnt.id]:
            self.vert_fun_global_keys.append(global_key("vert", vert_idx=k))

    def generate_edge_fun_global_keys(self) -> None:
        """Generate global keys for edge and vertex functions"""
        self.edge_fun_global_keys = []
        for k in range(self.num_edge_funs):
            self.edge_fun_global_keys.append(
                global_key("edge", edge_idx=self.e.id, edge_space_idx=k)
            )

    def build_spanning_set(self) -> None:
        """
        Spanning set of P_p(e) using traces of
                L_m(x_1) * L_n(x_2) - a * ell_0 - b * ell_1
        where
                L_j is the jth integrated Legendre polynomial,
                a = L_m(y_1) * L_n(y_2),
                b = L_m(z_1) * L_n(z_2),
                y = (y_1, y_2) starting vertex of edge
                z = (z_1, z_2) ending vertex of edge
        """

        # compute Legendre tensor products
        self.edge_fun_traces = integrated_legendre_tensor_products(self.deg)

        linear_fun_idx = [0, 1, self.deg + 1]

        # redo linear functions to avoid constants
        self.edge_fun_traces[linear_fun_idx[0]] = polynomial(
            [
                # 1 + x + y
                (1.0, 0, 0),
                (1.0, 1, 0),
                (1.0, 0, 1),
            ]
        )
        self.edge_fun_traces[linear_fun_idx[1]] = polynomial(
            [
                # 1 - x - y
                (+1.0, 0, 0),
                (-1.0, 1, 0),
                (-1.0, 0, 1),
            ]
        )
        self.edge_fun_traces[linear_fun_idx[2]] = polynomial(
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

            # correct edge functions to vanish at endpoints
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

                # TODO: find out why this is necessary
                # self.edge_fun_traces[j] *= -1

            # set vertex functions to barycentric coordinates
            self.vert_fun_traces.append(ell[0])
            self.vert_fun_traces.append(ell[1])

            # delete other linear functions from edge function list
            del self.edge_fun_traces[linear_fun_idx[2]]
            del self.edge_fun_traces[linear_fun_idx[1]]
            del self.edge_fun_traces[linear_fun_idx[0]]

            # add linear function vanishing at endpoints to edge function list
            self.edge_fun_traces.append(ell[2])

    def transform_coordinates_to_bounding_box(self) -> None:
        # get bounding box
        xmin, xmax, ymin, ymax = self.e.get_bounding_box()

        # scaling factors
        sx = (xmax - xmin) / 2
        sy = (ymax - ymin) / 2

        # translation factors
        tx = (xmax + xmin) / 2
        ty = (ymax + ymin) / 2

        # define affine change of coordinates
        qx = polynomial(
            [
                (-tx / sx, 0, 0),
                (1 / sx, 1, 0),
            ]
        )

        qy = polynomial(
            [
                (-ty / sy, 0, 0),
                (1 / sy, 0, 1),
            ]
        )

        # map Legendre tensor products from square to bounding box
        for j, f in enumerate(self.edge_fun_traces):
            self.edge_fun_traces[j] = f.compose(qx, qy)

    def reduce_to_basis(self) -> None:
        M = self.get_gram_matrix()
        idx = self.get_basis_index_set(M)
        basis = []
        for k in idx:
            basis.append(self.edge_fun_traces[k])
        self.edge_fun_traces = basis

    def get_gram_matrix(self, precond: bool = True) -> np.ndarray:
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
        m = np.shape(M)[0]
        for i in range(m):
            for j in range(i + 1, m):
                M[i, j] /= np.sqrt(M[i, i] * M[j, j])
                M[j, i] = M[i, j]
        for i in range(m):
            M[i, i] = 1
        return M

    def eliminate_zeros(self, M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
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
        idx: list[int] = []
        if np.shape(M)[0] == 0:
            return idx
        k = int(np.argmax(np.diag(M)))
        while M[k, k] > tol:
            idx.append(k)
            M -= np.outer(M[:, k], M[:, k]) / M[k, k]
            k = int(np.argmax(np.diag(M)))
        return sorted(idx)
