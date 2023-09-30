"""
nystrom.py
==========

Module containing the nystrom_solver class, which is used to represent a
Nyström solver for a given mesh cell K.
"""


import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from ..mesh.cell import cell
from ..mesh.closed_contour import closed_contour
from ..mesh.edge import edge
from ..mesh.quad import quad
from .d2n import log_terms
from .d2n.trace2tangential import get_weighted_tangential_derivative_from_trace


class nystrom_solver:
    """
    Nyström solver for a given mesh cell K. The Nyström solver is used to solve
    the Neumann problem on K, which can be used to compute harmonic conjugates
    of functions on K, for example.
    """

    # TODO add batch processing for multiple locfuns
    # TODO use multiprocessing to speed up computation

    K: cell
    single_layer_mat: np.ndarray
    double_layer_mat: np.ndarray
    single_layer_op: LinearOperator
    double_layer_op: LinearOperator
    double_layer_sum: np.ndarray
    lam_trace: np.ndarray
    lam_x1_trace: np.ndarray
    lam_x2_trace: np.ndarray
    dlam_dt_wgt: np.ndarray
    dlam_dn_wgt: np.ndarray
    T1_dlam_dt: np.ndarray
    Sn_lam: np.ndarray

    def __init__(self, K: cell, verbose: bool = False) -> None:
        """
        Constructor for Nyström solver. This constructor computes the single
        and double layer operators, as well as the logarithmic terms (if K has
        holes).

        Parameters
        ----------
        K : cell
            Mesh cell
        verbose : bool, optional
            Whether to print information about the Nyström solver, by default
            False
        """
        if verbose:
            msg = (
                "Setting up Nyström solver... "
                + f"{K.num_pts} sampled points on {K.num_edges} edge"
            )
            if K.num_edges > 1:
                msg += "s"
            print(msg)
        if not isinstance(K, cell):
            raise TypeError("K must be a cell")
        self.K = K
        self.build_single_layer_mat()
        self.build_double_layer_mat()
        self.build_single_and_double_layer_ops()
        if self.K.num_holes > 0:
            self.compute_log_terms()

    def build_single_and_double_layer_ops(self) -> None:
        """
        Build linear operator objects for the single and double layer
        operators.
        """
        self.double_layer_sum = np.sum(self.double_layer_mat, 1)
        self.single_layer_op = LinearOperator(
            dtype=float,
            shape=(self.K.num_pts, self.K.num_pts),
            matvec=self.linop4singlelayer,
        )
        self.double_layer_op = LinearOperator(
            dtype=float,
            shape=(self.K.num_pts, self.K.num_pts),
            matvec=self.linop4doublelayer,
        )

    # SOLVERS ################################################################

    def solve_neumann_zero_average(self, u_wnd: np.ndarray) -> np.ndarray:
        """Solve the Neumann problem with zero average on the boundary"""

        # RHS
        b = self.single_layer_op(u_wnd)

        # define linear operator for Neumann problem
        def A_fun(u: np.ndarray) -> np.ndarray:
            y = self.double_layer_op(u)
            y += self.K.integrate_over_boundary(u)
            return y

        # build linear operator object
        A = LinearOperator(
            dtype=float, shape=(self.K.num_pts, self.K.num_pts), matvec=A_fun
        )

        # solve Nystrom system using GMRES
        u, flag = gmres(A, b, atol=1e-12, tol=1e-12)

        # check for convergence
        if flag > 0:
            print(f"Something went wrong: GMRES returned flag = {flag}")

        return u

    def get_harmonic_conjugate(
        self, phi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Obtain a harmonic conjugate of phi on K"""

        # weighted tangential derivative of phi
        phi_wtd = get_weighted_tangential_derivative_from_trace(self.K, phi)

        # simply/multiply connected cases handled separately
        if self.K.num_holes == 0:  # simply connected
            return self.solve_neumann_zero_average(-1 * phi_wtd), np.zeros((0,))
        if self.K.num_holes > 0:  # multiply connected
            return self.get_harmonic_conjugate_multiply_connected(phi, phi_wtd)
        raise Exception("K.num_holes < 0")

    def get_harmonic_conjugate_multiply_connected(
        self, phi: np.ndarray, dphi_dt_wgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the Neumann problem for a harmonic conjugate of phi if K is
        multiply connected.
        """
        # array sizes
        N = self.K.num_pts
        m = self.K.num_holes

        # block RHS
        b = np.zeros((N + m,))
        b[:N] = self.single_layer_op(-dphi_dt_wgt)
        b[N:] = self.Sn(phi)

        # define linear operator for harmonic conjugate
        def linop4harmconj(x: np.ndarray) -> np.ndarray:
            psi_hat = x[:N]
            a = x[N:]
            y = np.zeros((N + m,))
            y[:N] = self.double_layer_op(psi_hat)
            y[:N] += self.K.integrate_over_boundary(psi_hat)
            y[:N] -= self.T1_dlam_dt @ a
            y[N:] = -self.St(psi_hat) + self.Sn_lam @ a
            return y

        # build linear operator object
        A = LinearOperator(
            dtype=float, shape=(N + m, N + m), matvec=linop4harmconj
        )

        # solve Nystrom system
        x, flag = gmres(A, b, atol=1e-12, tol=1e-12)
        psi_hat = x[:N]
        a = x[N:]

        # check for convergence
        if flag > 0:
            print(f"Something went wrong: GMRES returned flag = {flag}")

        return psi_hat, a

    # LOGARITHMIC TERMS #####################################################
    def compute_log_terms(self) -> None:
        """
        Compute and store logarithmic terms for multiply connected domains.
        """

        # traces and gradients of logarithmic corrections
        self.lam_trace = log_terms.get_log_trace(self.K)
        self.lam_x1_trace, self.lam_x2_trace = log_terms.get_log_grad(self.K)

        # tangential and normal derivatives of logarithmic terms
        self.dlam_dt_wgt = log_terms.get_dlam_dt_wgt(
            self.K, self.lam_x1_trace, self.lam_x2_trace
        )
        self.dlam_dn_wgt = log_terms.get_dlam_dn_wgt(
            self.K, self.lam_x1_trace, self.lam_x2_trace
        )

        # single layer operator applied to tangential derivatives of log terms
        self.T1_dlam_dt = np.zeros((self.K.num_pts, self.K.num_holes))
        for i in range(self.K.num_holes):
            self.T1_dlam_dt[:, i] = self.single_layer_op(self.dlam_dt_wgt[:, i])

        # H1 seminorms of logarithmic terms
        self.Sn_lam = np.zeros((self.K.num_holes, self.K.num_holes))
        for i in range(self.K.num_holes):
            self.Sn_lam[:, i] = self.Sn(self.lam_trace[:, i])

    def Su(self, vals: np.ndarray, dlam_du_wgt: np.ndarray) -> np.ndarray:
        """
        Abstraction of the operator
            S_u = int_{dK} u (d_lambda / du) ds
        where u is a vector field defined on the boundary of K.
        """
        out = np.zeros((self.K.num_holes,))
        for i in range(self.K.num_holes):
            out[i] = self.K.integrate_over_boundary_preweighted(
                vals * dlam_du_wgt[:, i]
            )
        return out

    def Sn(self, vals: np.ndarray) -> np.ndarray:
        """
        The operator
            S_n = int_{dK} (d_lambda / dn) vals ds
        where n is the outward normal.
        """
        return self.Su(vals, self.dlam_dn_wgt)

    def St(self, vals: np.ndarray) -> np.ndarray:
        """
        The operator
            S_n = int_{dK} (d_lambda / dt) vals ds
        where t is the unit tangent.
        """
        return self.Su(vals, self.dlam_dt_wgt)

    # SINGLE LAYER OPERATOR ###################################################

    def linop4singlelayer(self, u: np.ndarray) -> np.ndarray:
        """
        The single layer operator applied to u.
        """
        return self.single_layer_mat @ u

    def build_single_layer_mat(self) -> None:
        """
        Construct the single layer operator matrix.
        """
        self.single_layer_mat = np.zeros((self.K.num_pts, self.K.num_pts))
        for i in range(self.K.num_holes + 1):
            ii1 = self.K.component_start_idx[i]
            ii2 = self.K.component_start_idx[i + 1]
            for j in range(self.K.num_holes + 1):
                jj1 = self.K.component_start_idx[j]
                jj2 = self.K.component_start_idx[j + 1]
                self.single_layer_mat[
                    ii1:ii2, jj1:jj2
                ] = self.single_layer_component_block(i, j)

    def single_layer_component_block(self, i: int, j: int) -> np.ndarray:
        """
        Block of the single layer operator matrix corresponding to the i-th
        and j-th components of the boundary of K.
        """
        B_comp = np.zeros(
            (self.K.components[i].num_pts, self.K.components[j].num_pts)
        )
        for k in range(self.K.components[i].num_edges):
            kk1 = self.K.components[i].vert_idx[k]
            kk2 = self.K.components[i].vert_idx[k + 1]
            e = self.K.components[i].edges[k]
            # Martensen quadrature
            nm = (self.K.components[i].edges[k].num_pts - 1) // 2
            qm = quad(qtype="mart", n=nm)
            for ell in range(self.K.components[j].num_edges):
                ll1 = self.K.components[j].vert_idx[ell]
                ll2 = self.K.components[j].vert_idx[ell + 1]
                f = self.K.components[j].edges[ell]
                B_comp[kk1:kk2, ll1:ll2] = self.single_layer_edge_block(
                    e, f, qm
                )
        return B_comp

    def single_layer_edge_block(self, e: edge, f: edge, qm: quad) -> np.ndarray:
        """
        Block of the single layer operator matrix corresponding to the edges e
        and f. The quadrature object qm is the Martensen quadrature.
        """

        # allocate block
        B_edge = np.zeros((e.num_pts - 1, f.num_pts - 1))

        # trapezoid weight: pi in integrand cancels
        h = -0.5 / (f.num_pts - 1)

        # adapt quadrature to accommodate both trapezoid and Kress
        if f.quad_type[0:5] == "kress":
            j_start = 1
        else:
            j_start = 0

        if e == f:  # Kress and Martensen
            for i in range(e.num_pts - 1):
                for j in range(j_start, f.num_pts - 1):
                    ij = abs(i - j)
                    if ij == 0:
                        B_edge[i, i] = 2 * np.log(e.dx_norm[i])
                    else:
                        xy = e.x[:, i] - f.x[:, j]
                        xy2 = np.dot(xy, xy)
                        B_edge[i, j] = np.log(xy2 / qm.t[ij])
                    B_edge[i, j] *= h
                    B_edge[i, j] += qm.wgt[ij]

        else:  # different edges: Kress only
            for i in range(e.num_pts - 1):
                for j in range(j_start, f.num_pts - 1):
                    xy = e.x[:, i] - f.x[:, j]
                    xy2 = np.dot(xy, xy)
                    B_edge[i, j] = np.log(xy2) * h

        return B_edge

    # DOUBLE LAYER OPERATOR ##################################################

    def linop4doublelayer(self, u: np.ndarray) -> np.ndarray:
        """
        Operator for the double layer potential applied to u.
        """
        corner_values = u[self.K.closest_vert_idx]
        res = 0.5 * (u - corner_values)
        res += self.double_layer_mat @ u
        res -= corner_values * self.double_layer_sum
        return res

    def build_double_layer_mat(self) -> None:
        """
        Construct the double layer operator matrix.
        """
        self.double_layer_mat = np.zeros((self.K.num_pts, self.K.num_pts))
        for i in range(self.K.num_holes + 1):
            ii1 = self.K.component_start_idx[i]
            ii2 = self.K.component_start_idx[i + 1]
            for j in range(self.K.num_holes + 1):
                jj1 = self.K.component_start_idx[j]
                jj2 = self.K.component_start_idx[j + 1]
                self.double_layer_mat[
                    ii1:ii2, jj1:jj2
                ] = self.double_layer_component_block(
                    self.K.components[i], self.K.components[j]
                )

    def double_layer_component_block(
        self, ci: closed_contour, cj: closed_contour
    ) -> np.ndarray:
        """
        Block of the double layer operator matrix corresponding to the i-th
        and j-th components of the boundary of K.
        """
        B_comp = np.zeros((ci.num_pts, cj.num_pts))
        for k in range(ci.num_edges):
            kk1 = ci.vert_idx[k]
            kk2 = ci.vert_idx[k + 1]
            for ell in range(cj.num_edges):
                ll1 = cj.vert_idx[ell]
                ll2 = cj.vert_idx[ell + 1]
                B_comp[kk1:kk2, ll1:ll2] = self.double_layer_edge_block(
                    ci.edges[k], cj.edges[ell]
                )
        return B_comp

    def double_layer_edge_block(self, e: edge, f: edge) -> np.ndarray:
        """
        Block of the double layer operator matrix corresponding to the edges e
        and f.
        """

        # allocate block
        B_edge = np.zeros((e.num_pts - 1, f.num_pts - 1))

        # trapezoid step size
        h = 1 / (f.num_pts - 1)

        # check if edges are the same edge
        same_edge = e == f

        # adapt quadrature to accommodate both trapezoid and Kress
        if f.quad_type[0:5] == "kress":
            j_start = 1
        else:
            j_start = 0

        #
        for i in range(e.num_pts - 1):
            for j in range(j_start, f.num_pts - 1):
                if same_edge and i == j:
                    B_edge[i, i] = 0.5 * e.curvature[i]
                else:
                    xy = e.x[:, i] - f.x[:, j]
                    xy2 = np.dot(xy, xy)
                    B_edge[i, j] = np.dot(xy, f.unit_normal[:, j]) / xy2

                B_edge[i, j] *= f.dx_norm[j] * h

        return B_edge
