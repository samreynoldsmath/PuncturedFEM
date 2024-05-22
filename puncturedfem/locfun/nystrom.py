"""
Nyström Solver for solving integral equations.

Classes
-------
NystromSolver
"""

from typing import Optional

import numba
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from ..mesh.cell import MeshCell
from ..mesh.closed_contour import ClosedContour
from ..mesh.edge import Edge
from ..mesh.quad import Quad
from . import precond
from .fft_deriv import get_weighted_tangential_derivative_from_trace
from .trace import DirichletTrace


class NystromSolver:
    """
    Nyström Solver for a given mesh cell K.

    The Nyström Solver is used to solve the Neumann problem on K, which can be
    used to compute harmonic conjugates of functions on K, for example.

    Attributes
    ----------
    K : MeshCell
        The mesh cell
    single_layer_mat : np.ndarray
        Single layer operator matrix
    double_layer_mat : np.ndarray
        Double layer operator matrix
    single_layer_op : LinearOperator
        Single layer operator
    double_layer_op : LinearOperator
        Double layer operator
    double_layer_sum : np.ndarray
        Sum of the double layer operator matrix
    lam_trace : list[DirichletTrace]
        Traces of the logarithmic terms
    T1_dlam_dt : np.ndarray
        Single layer operator applied to the tangential derivatives of the
        logarithmic terms
    Sn_lam : np.ndarray
        H1 seminorms of the logarithmic terms
    A_simple : LinearOperator
        Linear operator for the Neumann problem
    A_augment : LinearOperator
        Linear operator for the multiply connected case
    precond_simple : LinearOperator
        Preconditioner for the Neumann problem
    precond_augment : LinearOperator
        Preconditioner for the multiply connected case
    antilap_strategty : str
        Strategy for computing the anti-Laplacian: "direct" or "fft"
    """

    K: MeshCell
    single_layer_mat: np.ndarray
    double_layer_mat: np.ndarray
    single_layer_op: LinearOperator
    double_layer_op: LinearOperator
    double_layer_sum: np.ndarray
    lam_trace: list[DirichletTrace]
    T1_dlam_dt: np.ndarray
    Sn_lam: np.ndarray
    A_simple: LinearOperator
    A_augment: LinearOperator
    precond_simple: LinearOperator
    precond_augment: LinearOperator
    antilap_strategy: str
    eta: Optional[list[DirichletTrace]] = None

    def __init__(
        self,
        K: MeshCell,
        antilap_strategy: str = "fft",
        precond_type: str = "jacobi",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Build the Nyström Solver.

        This constructor computes the single and double layer operators, as well
        as the logarithmic terms (if K has holes).

        Parameters
        ----------
        K : MeshCell
            Mesh MeshCell
        antilap_strategy : str, optional
            Strategy for computing the anti-Laplacian, by default "direct"
        precond_type : str, optional
            Preconditioner type, by default "jacobi"
        verbose : bool, optional
            Whether to print information about the Nyström Solver, by default
            False
        debug : bool, optional
            Whether to print debug information, by default False
        """
        # set mesh cell
        self.set_K(K)

        # print setup message
        if verbose or debug:
            print(self._setup_message())

        # build single and double layer operators
        if verbose:
            print("\tBuilding single and double layer operators...")
        self.build_single_layer_mat()
        self.build_double_layer_mat()
        self.build_single_and_double_layer_ops()

        # set up operator for Neumann problem
        self.A_simple = self._solve_neumann_zero_average_operator()

        # set up operator for multiply connected case
        if self.K.num_holes > 0:
            if verbose:
                print("\tComputing logarithmic terms...")
            self.compute_log_terms()
            self.A_augment = self._multiply_connected_operator()

        # build preconditioners
        if verbose:
            print(f"\tUsing preconditioner: {precond_type}")
        self.build_preconditioners(precond_type)

        # compute harmonic normal indicators
        if verbose:
            print(f"\tUsing biharmonic strategy: {antilap_strategy}")
        self._set_antilap_strategy(antilap_strategy)
        if self.K.num_holes > 0 and self.antilap_strategy == "fft":
            self._compute_harmonic_normal_indicators()

        # print condition number
        if debug:
            print("\tdebug-NystromSolver: Computing condition number...")
            kappa = self._get_operator_condition_number(
                self.precond_simple @ self.A_simple
            )
            print(f"\tdebug-NystromSolver: Condition number = {kappa:.2e}")
            if self.K.num_holes > 0:
                kappa = self._get_operator_condition_number(
                    self.precond_augment @ self.A_augment
                )
                print(
                    "\tdebug-NystromSolver: "
                    + f"Augmented condition number = {kappa:.2e}"
                )

        if verbose:
            print("\tNyström solver setup complete")

    def _setup_message(self) -> str:
        msg = (
            "Setting up Nyström solver... "
            + f"{self.K.num_pts} sampled points on {self.K.num_edges} edge"
        )
        if self.K.num_edges > 1:
            msg += "s"
        return msg

    def _set_antilap_strategy(self, antilap_strategy: str) -> None:
        if antilap_strategy not in ["direct", "fft"]:
            raise ValueError(
                f"'{antilap_strategy}'"
                + " is not a valid strategy, must be 'direct' or 'fft'"
            )
        self.antilap_strategy = antilap_strategy

    def set_K(self, K: MeshCell) -> None:
        """
        Set the MeshCell.

        Parameters
        ----------
        K : MeshCell
            Mesh MeshCell
        """
        if not isinstance(K, MeshCell):
            raise TypeError("K must be a MeshCell")
        self.K = K

    # SOLVERS ################################################################

    def solve_neumann_zero_average(self, u_wnd: np.ndarray) -> np.ndarray:
        """
        Solve the Neumann problem with zero average on the boundary.

        Parameters
        ----------
        u_wnd : np.ndarray
            Right-hand side of the Neumann problem.

        Returns
        -------
        np.ndarray
            Solution to the Neumann problem.
        """
        # right-hand side
        b = self.single_layer_op(u_wnd)

        # solve Nystrom system using GMRES
        return self._gmres_solve(A=self.A_simple, b=b, M=self.precond_simple)

    def get_harmonic_conjugate(
        self, phi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Obtain a harmonic conjugate of phi.

        In the case where K is multiply connected, phi is decomposed into a
        real part of a complex analytic function psi (which has a harmonic
        conjugate psi_hat) and a vector a of coefficients for the logarithmic
        terms.

        In the simply connected case, psi_hat is the harmonic conjugate of phi,
        and a is an empty array.

        Parameters
        ----------
        phi : np.ndarray
            Trace of a harmonic function on the boundary of a mesh cell.

        Returns
        -------
        psi_hat : np.ndarray
            Trace of the harmonic conjugate of the conjugable part of phi.
        a : np.ndarray
            Coefficients for the logarithmic terms. Empty array if K is simply
            connected.
        """
        # weighted tangential derivative of phi
        phi_wtd = get_weighted_tangential_derivative_from_trace(self.K, phi)

        # simply/multiply connected cases handled separately
        if self.K.num_holes == 0:  # simply connected
            return self.solve_neumann_zero_average(-1 * phi_wtd), np.zeros((0,))
        if self.K.num_holes > 0:  # multiply connected
            return self._get_harmonic_conjugate_multiply_connected(phi, phi_wtd)
        raise ValueError("K.num_holes < 0")

    def _get_harmonic_conjugate_multiply_connected(
        self, phi: np.ndarray, dphi_dt_wgt: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # array sizes
        N = self.K.num_pts
        m = self.K.num_holes

        # block RHS
        b = np.zeros((N + m,))
        b[:N] = self.single_layer_op(-dphi_dt_wgt)
        b[N:] = self.Sn(phi)

        # solve Nystrom system with GMRES
        x = self._gmres_solve(A=self.A_augment, b=b, M=self.precond_augment)
        psi_hat = x[:N]
        a = x[N:]

        return psi_hat, a

    def _gmres_solve(
        self, A: LinearOperator, b: np.ndarray, M: LinearOperator
    ) -> np.ndarray:
        x, flag = gmres(A=A, b=b, M=M, atol=1e-12, rtol=1e-12)
        if flag > 0:
            r = b - A @ x
            print(
                "warn-NystromSolver: "
                + f"GMRES failed to converge after {flag} iterations"
                + f", residual norm = {np.linalg.norm(r):.2e}"
            )
        return x

    # OPERATORS ###############################################################

    def build_single_and_double_layer_ops(self) -> None:
        """Build single and double layer operators."""
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

    def _solve_neumann_zero_average_operator(self) -> LinearOperator:
        # define linear operator for Neumann problem
        def A_fun(u: np.ndarray) -> np.ndarray:
            y = self.double_layer_op(u)
            y += self.K.integrate_over_boundary(u)
            return y

        # build linear operator object
        return LinearOperator(
            dtype=float,
            shape=(self.K.num_pts, self.K.num_pts),
            matvec=A_fun,
        )

    def _multiply_connected_operator(self) -> LinearOperator:
        # array sizes
        N = self.K.num_pts
        m = self.K.num_holes

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
        return LinearOperator(
            dtype=float, shape=(N + m, N + m), matvec=linop4harmconj
        )

    # PRECONDITIONERS ########################################################

    def build_preconditioners(self, precond_type: str = "jacobi") -> None:
        """
        Build a preconditioner for the Neumann problem.

        Parameters
        ----------
        precond_type : Optional[str]
            Preconditioner type, by default "jacobi"

        Notes
        -----
        Available preconditioners:
        - "jacobi": Jacobi preconditioner
        """
        if precond_type == "jacobi":
            self.precond_simple = precond.jacobi_preconditioner(self.A_simple)
            if self.K.num_holes > 0:
                self.precond_augment = precond.jacobi_preconditioner(
                    self.A_augment
                )
        else:
            raise ValueError("Invalid preconditioner type")

    # LOGARITHMIC TERMS #####################################################
    def compute_log_terms(self) -> None:
        """
        Compute and store logarithmic terms for multiply connected domains.
        """
        x1, x2 = self.K.get_boundary_points()
        self.lam_trace = []
        self.T1_dlam_dt = np.zeros((self.K.num_pts, self.K.num_holes))
        self.Sn_lam = np.zeros((self.K.num_holes, self.K.num_holes))

        for j in range(self.K.num_holes):
            xi = self.K.components[j + 1].interior_point
            x1_xi = x1 - xi.x
            x2_xi = x2 - xi.y
            x_xi_square_norm = x1_xi**2 + x2_xi**2

            # trace of logarithmic term
            lam_j = DirichletTrace(
                edges=self.K.get_edges(), values=0.5 * np.log(x_xi_square_norm)
            )

            # gradient of logarithmic term
            lam_j_xi = x1_xi / x_xi_square_norm
            lam_j_x2 = x2_xi / x_xi_square_norm

            # weighted normal derivative
            wnd = self.K.dot_with_normal(lam_j_xi, lam_j_x2)
            wnd = self.K.multiply_by_dx_norm(wnd)
            lam_j.set_weighted_normal_derivative(wnd)

            # weighted tangential derivative
            wtd = self.K.dot_with_tangent(lam_j_xi, lam_j_x2)
            wtd = self.K.multiply_by_dx_norm(wtd)
            lam_j.set_weighted_tangential_derivative(wtd)

            self.lam_trace.append(lam_j)

        # single layer operator applied to tangential derivatives of log terms
        for j in range(self.K.num_holes):
            lam_j_wtd = self.lam_trace[j].w_tang_deriv
            self.T1_dlam_dt[:, j] = self.single_layer_op(lam_j_wtd)

        # H1 seminorms of logarithmic terms
        for j in range(self.K.num_holes):
            vals = self.lam_trace[j].values
            self.Sn_lam[:, j] = self.Sn(vals)

    def Sn(self, vals: np.ndarray) -> np.ndarray:
        """
        Apply the operator S_n(vals) = int_{dK} vals * d_lambda/dn ds.

        Parameters
        ----------
        vals : np.ndarray
            Vector vals

        Returns
        -------
        np.ndarray
            Result of the operator S_n applied to vals.
        """
        res = np.zeros((self.K.num_holes,))
        for j in range(self.K.num_holes):
            res[j] = self.K.integrate_over_boundary_preweighted(
                vals * self.lam_trace[j].w_norm_deriv
            )
        return res

    def St(self, vals: np.ndarray) -> np.ndarray:
        """
        Apply the operator S_n(vals) = int_{dK} vals * d_lambda / dt ds.

        Parameters
        ----------
        vals : np.ndarray
            Vector vals

        Returns
        -------
        np.ndarray
            Result of the operator S_t applied to vals.
        """
        res = np.zeros((self.K.num_holes,))
        for j in range(self.K.num_holes):
            res[j] = self.K.integrate_over_boundary_preweighted(
                vals * self.lam_trace[j].w_tang_deriv
            )
        return res

    # HARMONIC NORMAL INDICATORS #############################################
    def _compute_harmonic_normal_indicators(self) -> None:
        self.eta = []
        c0 = self.K.components[0]
        outer_arc_length = c0.integrate_over_closed_contour(
            np.ones((c0.num_pts,))
        )
        num_outer_edges = c0.num_edges
        edge_idx = num_outer_edges
        for c in self.K.components[1:]:
            chi_trace_j = DirichletTrace(edges=self.K.get_edges(), values=0.0)
            inner_arc_length = c.integrate_over_closed_contour(
                np.ones((c.num_pts,))
            )
            for k in range(num_outer_edges):
                chi_trace_j.set_trace_values_on_edge(
                    edge_index=k,
                    values=-inner_arc_length / outer_arc_length,
                )
            for k in range(c.num_edges):
                chi_trace_j.set_trace_values_on_edge(
                    edge_index=edge_idx + k, values=1.0
                )
            edge_idx += c.num_edges
            eta_wnd_j = self.K.multiply_by_dx_norm(chi_trace_j.values)
            eta_j = self.solve_neumann_zero_average(eta_wnd_j)
            self.eta.append(
                DirichletTrace(edges=self.K.get_edges(), values=eta_j)
            )

    # SINGLE LAYER OPERATOR ###################################################

    def linop4singlelayer(self, u: np.ndarray) -> np.ndarray:
        """
        Apply the single layer operator to u.

        Parameters
        ----------
        u : np.ndarray
            Vector u

        Returns
        -------
        np.ndarray
            Result of the single layer operator applied to u.
        """
        return self.single_layer_mat @ u

    def build_single_layer_mat(self) -> None:
        """Construct the single layer operator matrix."""
        self.single_layer_mat = np.zeros((self.K.num_pts, self.K.num_pts))
        for i in range(self.K.num_holes + 1):
            ii1 = self.K.component_start_idx[i]
            ii2 = self.K.component_start_idx[i + 1]
            for j in range(self.K.num_holes + 1):
                jj1 = self.K.component_start_idx[j]
                jj2 = self.K.component_start_idx[j + 1]
                self.single_layer_mat[ii1:ii2, jj1:jj2] = (
                    self.single_layer_component_block(i, j)
                )

    def single_layer_component_block(self, i: int, j: int) -> np.ndarray:
        """
        Block of the single layer operator matrix by components.

        Parameters
        ----------
        i : int
            Component i
        j : int
            Component j

        Returns
        -------
        np.ndarray
            Block of the single layer operator matrix for the components i and
            j.
        """
        B_comp = np.zeros(
            (
                self.K.components[i].num_pts,
                self.K.components[j].num_pts,
            )
        )
        for k in range(self.K.components[i].num_edges):
            kk1 = self.K.components[i].vert_idx[k]
            kk2 = self.K.components[i].vert_idx[k + 1]
            e = self.K.components[i].edges[k]
            # Martensen Quadrature
            nm = (self.K.components[i].edges[k].num_pts - 1) // 2
            qm = Quad(qtype="mart", n=nm)
            for ell in range(self.K.components[j].num_edges):
                ll1 = self.K.components[j].vert_idx[ell]
                ll2 = self.K.components[j].vert_idx[ell + 1]
                f = self.K.components[j].edges[ell]
                B_comp[kk1:kk2, ll1:ll2] = self.single_layer_edge_block(
                    e, f, qm
                )
        return B_comp

    def single_layer_edge_block(self, e: Edge, f: Edge, qm: Quad) -> np.ndarray:
        """
        Block of the single layer operator matrix corresponding by edges.

        Parameters
        ----------
        e : Edge
            Edge e
        f : Edge
            Edge f

        Returns
        -------
        np.ndarray
            Block of the single layer operator matrix for the edges e and f.
        """
        # allocate block
        # B_edge = np.zeros((e.num_pts - 1, f.num_pts - 1))

        # trapezoid weight: pi in integrand cancels
        h = -0.5 / (f.num_pts - 1)

        # adapt Quadrature to accommodate both trapezoid and Kress
        if f.quad_type[0:5] == "kress":
            j_start = 1
        else:
            j_start = 0

        if e == f:  # Kress and Martensen
            B_edge = _single_layer_same_edge_block(
                e.num_pts,
                j_start,
                h,
                e.x,
                e.dx_norm,
                qm.wgt,
                qm.t,
            )

        else:  # different edges: Kress only
            B_edge = _single_layer_distinct_edge_block(
                e.num_pts, f.num_pts, h, e.x, f.x, j_start
            )

        # raise exception when non-numeric value encountered
        if np.isnan(B_edge).any() or np.isinf(B_edge).any():
            raise ZeroDivisionError("Nystrom system could not be constructed")

        return B_edge

    # DOUBLE LAYER OPERATOR ##################################################

    def linop4doublelayer(self, u: np.ndarray) -> np.ndarray:
        """
        Apply the operator for the double layer potential to u.

        Parameters
        ----------
        u : np.ndarray
            Vector u

        Returns
        -------
        np.ndarray
            Result of the double layer operator applied to u.
        """
        corner_values = u[self.K.closest_vert_idx]
        res = 0.5 * (u - corner_values)
        res += self.double_layer_mat @ u
        res -= corner_values * self.double_layer_sum
        return res

    def build_double_layer_mat(self) -> None:
        """Construct the double layer operator matrix."""
        self.double_layer_mat = np.zeros((self.K.num_pts, self.K.num_pts))
        for i in range(self.K.num_holes + 1):
            ii1 = self.K.component_start_idx[i]
            ii2 = self.K.component_start_idx[i + 1]
            for j in range(self.K.num_holes + 1):
                jj1 = self.K.component_start_idx[j]
                jj2 = self.K.component_start_idx[j + 1]
                self.double_layer_mat[ii1:ii2, jj1:jj2] = (
                    self.double_layer_component_block(
                        self.K.components[i], self.K.components[j]
                    )
                )

    def double_layer_component_block(
        self, ci: ClosedContour, cj: ClosedContour
    ) -> np.ndarray:
        """
        Get the block of the double layer operator matrix by components.

        Parameters
        ----------
        ci : ClosedContour
            Component ci
        cj : ClosedContour
            Component cj

        Returns
        -------
        np.ndarray
            Block of the double layer operator matrix for the components ci and
            cj.
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

    def double_layer_edge_block(self, e: Edge, f: Edge) -> np.ndarray:
        """
        Get the block of the double layer operator matrix by edges.

        Parameters
        ----------
        e : Edge
            Edge e
        f : Edge
            Edge f

        Returns
        -------
        np.ndarray
            Block of the double layer operator matrix for the edges e and f.
        """
        # trapezoid step size
        h = 1 / (f.num_pts - 1)

        # adapt Quadrature to accommodate both trapezoid and Kress
        if f.quad_type[0:5] == "kress":
            j_start = 1
        else:
            j_start = 0

        # compute entries
        if e == f:
            B_edge = _double_layer_same_edge_block(
                e.num_pts,
                h,
                e.x,
                j_start,
                e.curvature,
                e.unit_normal,
                e.dx_norm,
            )
        else:
            B_edge = _double_layer_distinct_edge_block(
                e.num_pts,
                f.num_pts,
                h,
                e.x,
                f.x,
                j_start,
                f.unit_normal,
                f.dx_norm,
            )

        # raise exception when non-numeric value encountered
        if np.isnan(B_edge).any() or np.isinf(B_edge).any():
            raise ZeroDivisionError("Nystrom system could not be constructed")

        return B_edge

    # DEBUGGING ###############################################################

    def _get_operator_matrix(self, A: LinearOperator) -> np.ndarray:
        I = np.eye(A.shape[0])
        A_mat = np.zeros(A.shape)
        for i in range(A.shape[0]):
            A_mat[:, i] = A @ I[:, i]
        return A_mat

    def _get_operator_condition_number(self, A: LinearOperator) -> float:
        A_mat = self._get_operator_matrix(A)
        return np.linalg.cond(A_mat)


@numba.njit
def _single_layer_same_edge_block(
    num_pts: int,
    j_start: int,
    h: float,
    x: np.ndarray,
    dx_norm: np.ndarray,
    mart_wgt: np.ndarray,
    mart_sin: np.ndarray,
) -> np.ndarray:
    B_edge = np.zeros((num_pts - 1, num_pts - 1))
    for i in range(num_pts - 1):
        for j in range(j_start, num_pts - 1):
            ij = abs(i - j)
            if mart_sin[ij] < 1e-14:
                if dx_norm[i] < 1e-14:
                    B_edge[i, i] = 0.0
                else:
                    B_edge[i, i] = 2 * np.log(dx_norm[j])
            else:
                xy = np.ascontiguousarray(x[:, i] - x[:, j])
                xy2 = np.dot(xy, xy)
                B_edge[i, j] = np.log(xy2 / mart_sin[ij])
            B_edge[i, j] *= h
            B_edge[i, j] += mart_wgt[ij]
    return B_edge


@numba.njit
def _single_layer_distinct_edge_block(
    e_num_pts: int,
    f_num_pts: int,
    h: float,
    e_x: np.ndarray,
    f_x: np.ndarray,
    j_start: int,
) -> np.ndarray:
    B_edge = np.zeros((e_num_pts - 1, f_num_pts - 1))
    for i in range(e_num_pts - 1):
        for j in range(j_start, f_num_pts - 1):
            xy = np.ascontiguousarray(e_x[:, i] - f_x[:, j])
            xy2 = np.dot(xy, xy)
            B_edge[i, j] = np.log(xy2) * h
    return B_edge


@numba.njit
def _double_layer_same_edge_block(
    num_pts: int,
    h: float,
    x: np.ndarray,
    j_start: int,
    curvature: np.ndarray,
    unit_normal: np.ndarray,
    dx_norm: np.ndarray,
) -> np.ndarray:
    B_edge = np.zeros((num_pts - 1, num_pts - 1))
    for i in range(num_pts - 1):
        for j in range(j_start, num_pts - 1):
            xy = np.ascontiguousarray(x[:, i] - x[:, j])
            xy2 = np.dot(xy, xy)
            if xy2 < 1e-12:
                B_edge[i, j] = 0.5 * curvature[j]
            else:
                t = np.ascontiguousarray(unit_normal[:, j])
                B_edge[i, j] = np.dot(xy, t) / xy2
            B_edge[i, j] *= dx_norm[j] * h
    return B_edge


@numba.njit
def _double_layer_distinct_edge_block(
    e_num_pts: int,
    f_num_pts: int,
    h: float,
    e_x: np.ndarray,
    f_x: np.ndarray,
    j_start: int,
    f_unit_normal: np.ndarray,
    f_dx_norm: np.ndarray,
) -> np.ndarray:
    B_edge = np.zeros((e_num_pts - 1, f_num_pts - 1))
    for i in range(e_num_pts - 1):
        for j in range(j_start, f_num_pts - 1):
            xy = np.ascontiguousarray(e_x[:, i] - f_x[:, j])
            xy2 = np.dot(xy, xy)
            t = np.ascontiguousarray(f_unit_normal[:, j])
            B_edge[i, j] = f_dx_norm[j] * h * np.dot(xy, t) / xy2
    return B_edge
