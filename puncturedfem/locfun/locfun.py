"""
Elements of the local Poisson space V_p(K).

Classes
-------
LocalFunction
"""

from __future__ import annotations

from typing import Optional, Union

from deprecated import deprecated
import numpy as np

from ..solver.globkey import GlobalKey
from ..util.types import FloatLike
from . import antilap, fft_deriv
from .nystrom import NystromSolver
from .poly.integrate_poly import integrate_poly_over_mesh_cell
from .poly.poly import Polynomial
from .trace import DirichletTrace


@deprecated(
    version="0.5.0",
    reason="LocalFunction is being deprecated in favor of the LocalPoissonFunction class.",
)
class LocalFunction:
    """
    Function with a polynomial Laplacian and continuous trace on the boundary.

    Typically, this class is used to represent functions in the local Poisson
    space V_p(K), which consists of functions whose Laplacian is a polynomial of
    degree <= p-2 on each edge of K, and whose continuous trace on any edge of
    the boundary of K is the restriction of a polynomial to that edge. However,
    it can also be used to represent functions with an arbitrary continuous
    trace on the boundary of K. Any such function v can be decomposed as
        v = P + phi
    where P is a polynomial of degree <= p in K and phi is a harmonic function.
    (Note that this decomposition is not unique, since there are harmonic
    polynomials). We refer to P as the "polynomial part" and phi as the
    "harmonic part". Furthermore, in multiply connected cells the harmonic part
    can be decomposed as
        phi = psi + sum_{j=1}^m a_j log |x - xi_j|
    where xi_j is a fixed arbitrary point in the jth hole of K, and psi is a
    harmonic function with a harmonic conjugate psi_hat. We refer to psi as the
    "conjugable part". Given a parametrization x(tau) of the boundary of K, we
    refer to
        (d / dtau) v(x(tau)) = nabla v(x(tau)) * x'(tau)
    as the "weighted normal derivative" of v, which we commonly abbreviate in
    code with "wnd". Similarly, the "weighted tangential derivative" is
    abbreviated "wtd". An "anti-Laplacian" of u is any function U such that
        Delta U = u
    where Delta is the Laplacian operator. In particular, we desire an anti-
    Laplacian of the harmonic part phi, as well as its weighted normal
    derivative. This allows us to compute the L^2 inner product of two local
    functions v and w using only boundary integrals. Interior values of a local
    function v, as well as its gradient, can be computed using Cauchy's integral
    formula.

    Attributes
    ----------
    nyst : NystromSolver
        Nystrom solver object for solving integral equations.
    poly_part : Polynomial
        Polynomial part P.
    poly_part_trace : DirichletTrace
        Trace of P, the polynomial part.
    harm_part_trace : DirichletTrace
        Trace of phi, the harmonic part.
    harm_conj_trace : DirichletTrace
        Trace of psi_hat, the harmonic conjugate of the conjugable part.
    log_coef : list[float]
        Logarithmic coefficients a_j of the harmonic part.
    biharmonic_trace : DirichletTrace
        Trace of Phi, an anti-Laplacian of the harmonic part.
    int_vals : numpy.ndarray
        Interior values of the local function.
    int_grad1 : numpy.ndarray
        First component of the gradient of the local function.
    int_grad2 : numpy.ndarray
        Second component of the gradient of the local function.
    key : GlobalKey
        A unique tag that identifies the local function in the global space.
    """

    nyst: NystromSolver
    poly_part: Polynomial
    poly_part_trace: DirichletTrace
    harm_part_trace: DirichletTrace
    harm_conj_trace: DirichletTrace
    log_coef: list[float]
    biharmonic_trace: Optional[DirichletTrace]
    int_vals: np.ndarray
    int_grad1: np.ndarray
    int_grad2: np.ndarray
    key: GlobalKey

    def __init__(
        self,
        nyst: NystromSolver,
        laplacian: Polynomial = Polynomial(),
        trace: Union[DirichletTrace, FloatLike] = 0,
        compute_for_l2: bool = True,
        compute_int_vals: bool = False,
        compute_int_grad: bool = False,
        key: Optional[GlobalKey] = None,
    ) -> None:
        """
        Build an element of the local Poisson space V_p(K).

        Parameters
        ----------
        nyst : NystromSolver
            Nystrom solver object for solving integral equations.
        laplacian : Polynomial, optional
            Polynomial Laplacian of the local function, by default Polynomial().
        trace : DirichletTrace or FloatLike, optional
            Dirichlet trace of the local function, by default 0. If a numpy
            array, must be the same length as the number of sampled points on
            the boundary.
        compute_for_l2 : bool, optional
            Whether to compute the biharmonic part, by default True. Disabling
            is useful when only the H^1 semi-inner product is needed.
        compute_int_vals : bool, optional
            Whether to compute the interior values, by default False.
        compute_int_grad : bool, optional
            Whether to compute the gradient, by default False.
        key : Optional[GlobalKey], optional
            A unique tag that identifies the local function in the global space.
        """
        self._set_key(key)
        self._set_nystrom_solver(nyst)
        self._compute_polynomial_part(laplacian)
        if isinstance(trace, (float, int, np.ndarray)):
            trace = DirichletTrace(edges=nyst.K.get_edges(), values=trace)
        self._compute_harmonic_part_trace(trace)
        self._compute_harmonic_conjugate()
        self._compute_harmonic_weighted_normal_derivative()
        if compute_for_l2:
            self._compute_biharmonic()
        if compute_int_vals or compute_int_grad:
            self.compute_interior_values(compute_int_grad)

    def get_h1_semi_inner_prod(self, other: LocalFunction) -> float:
        """
        Return the H^1 semi-inner product int_K grad(self) * grad(other) dx.

        Parameters
        ----------
        other : LocalFunction
            Another LocalFunction on the same mesh cell.
        """
        if not isinstance(other, LocalFunction):
            raise TypeError("other must be aLocalFunction")

        # grad P * grad Q
        Px, Py = self.poly_part.grad()
        Qx, Qy = other.poly_part.grad()
        gradP_gradQ = Px * Qx + Py * Qy
        val = integrate_poly_over_mesh_cell(gradP_gradQ, self.nyst.K)

        # grad phi * grad Q
        val += self.nyst.K.integrate_over_boundary_preweighted(
            self.harm_part_trace.w_norm_deriv * other.poly_part_trace.values
        )

        # grad P * grad psi
        val += self.nyst.K.integrate_over_boundary_preweighted(
            self.poly_part_trace.values * other.harm_part_trace.w_norm_deriv
        )

        # grad phi * grad psi
        val += self.nyst.K.integrate_over_boundary_preweighted(
            self.harm_part_trace.w_norm_deriv * other.harm_part_trace.values
        )

        return val

    def get_l2_inner_prod(self, other: LocalFunction) -> float:
        """
        Return the L^2 inner product int_K (self) * (other) dx.

        Parameters
        ----------
        other : LocalFunction
            Another LocalFunction on the same mesh cell.
        """
        if not isinstance(other, LocalFunction):
            raise TypeError("other must be aLocalFunction")
        if self.biharmonic_trace is None or other.biharmonic_trace is None:
            raise ValueError("Biharmonic trace not computed.")

        # sampled points on the boundary
        x1, x2 = self.nyst.K.get_boundary_points()

        # P * Q
        PQ = self.poly_part * other.poly_part
        val = integrate_poly_over_mesh_cell(PQ, self.nyst.K)

        # phi * Q
        R = other.poly_part.anti_laplacian()
        R_trace = R(x1, x2)
        R_wnd = R.get_weighted_normal_derivative(self.nyst.K)
        val += self.nyst.K.integrate_over_boundary_preweighted(
            self.harm_part_trace.values * R_wnd
        )
        val -= self.nyst.K.integrate_over_boundary_preweighted(
            self.harm_part_trace.w_norm_deriv * R_trace
        )

        # P * psi
        R = self.poly_part.anti_laplacian()
        R_trace = R(x1, x2)
        R_wnd = R.get_weighted_normal_derivative(self.nyst.K)
        val += self.nyst.K.integrate_over_boundary_preweighted(
            other.harm_part_trace.values * R_wnd
        )
        val -= self.nyst.K.integrate_over_boundary_preweighted(
            other.harm_part_trace.w_norm_deriv * R_trace
        )

        # phi * psi
        val += self.nyst.K.integrate_over_boundary_preweighted(
            self.biharmonic_trace.w_norm_deriv * other.harm_part_trace.values
        )
        val -= self.nyst.K.integrate_over_boundary_preweighted(
            self.biharmonic_trace.values * other.harm_part_trace.w_norm_deriv
        )

        return val

    def _set_key(self, key: Optional[GlobalKey]) -> None:
        if key is None:
            return
        if not isinstance(key, GlobalKey):
            raise TypeError("key must be a GlobalKey")
        self.key = key

    def _set_nystrom_solver(self, nyst: NystromSolver) -> None:
        if not isinstance(nyst, NystromSolver):
            raise TypeError("nyst must be a NystromSolver")
        self.nyst = nyst

    def _compute_polynomial_part(self, laplacian: Polynomial) -> None:
        # polynomial part and an anti-Laplacian
        self.poly_part = laplacian.anti_laplacian()
        self.poly_part_anti_lap = self.poly_part.anti_laplacian()

        # trace of the polynomial part
        self.poly_part_trace = DirichletTrace(
            edges=self.nyst.K.get_edges(), funcs=self.poly_part
        )

        # weighted normal derivative of the polynomial part
        x1, x2 = self.nyst.K.get_boundary_points()
        g1, g2 = self.poly_part.grad()
        P_nd = self.nyst.K.dot_with_normal(g1(x1, x2), g2(x1, x2))
        P_wnd = self.nyst.K.multiply_by_dx_norm(P_nd)
        self.poly_part_trace.set_weighted_normal_derivative(P_wnd)

    def _compute_harmonic_part_trace(self, trace: DirichletTrace) -> None:
        self.harm_part_trace = DirichletTrace(
            edges=self.nyst.K.get_edges(),
            values=trace.values - self.poly_part_trace.values,
        )

    def _compute_harmonic_conjugate(self) -> None:
        harm_conj_trace_values, log_coef = self.nyst.get_harmonic_conjugate(
            phi=self.harm_part_trace.values
        )
        self.harm_conj_trace = DirichletTrace(
            edges=self.nyst.K.get_edges(), values=harm_conj_trace_values
        )
        self.log_coef = list(log_coef)

    def _compute_harmonic_weighted_normal_derivative(self) -> None:
        harm_part_wnd = (
            fft_deriv.get_weighted_tangential_derivative_from_trace(
                self.nyst.K, self.harm_conj_trace.values
            )
        )
        for j in range(self.nyst.K.num_holes):
            harm_part_wnd += (
                self.log_coef[j] * self.nyst.lam_trace[j].w_norm_deriv
            )
        self.harm_part_trace.set_weighted_normal_derivative(harm_part_wnd)

    def _get_conjugable_part(self) -> np.ndarray:
        lam = np.zeros((self.nyst.K.num_pts,))
        for j in range(self.nyst.K.num_holes):
            lam += self.log_coef[j] * self.nyst.lam_trace[j].values
        return self.harm_part_trace.values - lam

    def _compute_biharmonic(self) -> None:
        psi = self._get_conjugable_part()
        psi_hat = self.harm_conj_trace.values
        (
            big_phi,
            big_phi_wnd,
        ) = antilap.get_anti_laplacian_harmonic(
            self.nyst, psi, psi_hat, np.array(self.log_coef)
        )
        self.biharmonic_trace = DirichletTrace(
            edges=self.nyst.K.get_edges(), values=big_phi
        )
        self.biharmonic_trace.set_weighted_normal_derivative(big_phi_wnd)

    def compute_interior_values(self, compute_int_grad: bool = True) -> None:
        """
        Compute the interior values.

        Also compute the components of the gradient if compute_int_grad is True.
        The interior values are stored in self.int_vals, and the gradient
        components are stored in self.int_grad1 and self.int_grad2.

        Parameters
        ----------
        compute_int_grad : bool, optional
            Whether or not to compute the gradient, by default True.
        """
        # points for evaluation
        y1 = self.nyst.K.int_x1[self.nyst.K.is_inside]
        y2 = self.nyst.K.int_x2[self.nyst.K.is_inside]

        # initialize temporary arrays
        N = len(y1)
        vals = np.zeros((N,))
        grad1 = np.zeros((N,))
        grad2 = np.zeros((N,))

        # polynomial part
        vals = self.poly_part(y1, y2)

        # gradient Polynomial part
        if compute_int_grad:
            Px, Py = self.poly_part.grad()
            grad1 = Px(y1, y2)
            grad2 = Py(y1, y2)

        # logarithmic part
        for k in range(self.nyst.K.num_holes):
            xi = self.nyst.K.components[k + 1].interior_point
            y_xi_1 = y1 - xi.x
            y_xi_2 = y2 - xi.y
            y_xi_norm_sq = y_xi_1**2 + y_xi_2**2
            vals += 0.5 * self.log_coef[k] * np.log(y_xi_norm_sq)
            if compute_int_grad:
                grad1 += self.log_coef[k] * y_xi_1 / y_xi_norm_sq
                grad2 += self.log_coef[k] * y_xi_2 / y_xi_norm_sq

        # conjugable part
        psi = self._get_conjugable_part()
        psi_hat = self.harm_conj_trace.values

        # boundary points
        bdy_x1, bdy_x2 = self.nyst.K.get_boundary_points()

        # shifted coordinates
        M = self.nyst.K.num_pts
        xy1 = np.reshape(bdy_x1, (1, M)) - np.reshape(y1, (N, 1))
        xy2 = np.reshape(bdy_x2, (1, M)) - np.reshape(y2, (N, 1))
        xy_norm_sq = xy1**2 + xy2**2

        # components of unit tangent vector
        t1, t2 = self.nyst.K.get_unit_tangent()

        # integrand for interior values
        eta = (xy1 * psi + xy2 * psi_hat) / xy_norm_sq
        eta_hat = (xy1 * psi_hat - xy2 * psi) / xy_norm_sq
        f = t1 * eta_hat + t2 * eta

        # integrands for gradient
        if compute_int_grad:
            omega = (xy1 * eta + xy2 * eta_hat) / xy_norm_sq
            omega_hat = (xy1 * eta_hat - xy2 * eta) / xy_norm_sq
            g1 = t1 * omega_hat + t2 * omega
            g2 = t1 * omega - t2 * omega_hat

        # Jacobian and trapezoid weights
        dx_norm = self.nyst.K.get_dx_norm()
        h = 2 * np.pi * self.nyst.K.num_edges / self.nyst.K.num_pts
        dx_norm *= h

        # interior values and gradient of conjugable part via Cauchy's
        # integral formula
        vals += np.sum(dx_norm * f, axis=1) * 0.5 / np.pi
        if compute_int_grad:
            grad1 += np.sum(dx_norm * g1, axis=1) * 0.5 / np.pi
            grad2 += np.sum(dx_norm * g2, axis=1) * 0.5 / np.pi

        # size of grid of evaluation points
        rows, cols = self.nyst.K.int_mesh_size

        # initialize arrays with proper size
        self.int_vals = np.empty((rows, cols))
        self.int_grad1 = np.empty((rows, cols))
        self.int_grad2 = np.empty((rows, cols))

        # default to not-a-number
        self.int_vals[:] = np.nan
        self.int_grad1[:] = np.nan
        self.int_grad2[:] = np.nan

        # set values within cell interior
        self.int_vals[self.nyst.K.is_inside] = vals
        self.int_grad1[self.nyst.K.is_inside] = grad1
        self.int_grad2[self.nyst.K.is_inside] = grad2
