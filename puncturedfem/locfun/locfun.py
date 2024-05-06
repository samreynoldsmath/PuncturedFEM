"""
Elements of the local Poisson space V_p(K).

Classes
-------
LocalFunction
"""

from typing import Optional

import numpy as np

from ..mesh.mesh_exceptions import SizeMismatchError
from ..solver.globkey import GlobalKey
from . import antilap, d2n
from .nystrom import NystromSolver
from .poly.integrate_poly import integrate_poly_over_mesh
from .poly.piecewise_poly import PiecewisePolynomial
from .poly.poly import Polynomial


class LocalFunction:
    """
    Element of the local Poisson space V_p(K).

    The trace is continuous and a Polynomial of degree <= p on each Edge of K,
    and whose Laplacian is a Polynomial of degree <= p-2 on each Edge of K. Can
    also be used to represent functions with an arbitrary continuous trace on
    the boundary of K. Any such function v can be decomposed as
        v = P + phi
    where P is a Polynomial of degree <= p in K and phi is a harmonic function.
    (Note that this decomposition is not unique, since there are harmonic
    Polynomials). We refer to P as the "Polynomial part" and phi as the
    "harmonic part". Furthermore, in multiply connected MeshCells the harmonic
    part can be decomposed as
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
    key : GlobalKey
        A unique tag that identifies the local function in the global space.
    trace : numpy.ndarray
        Dirichlet trace values on the boundary of K.
    poly_trace : PiecewisePolynomial
        Piecewise Polynomial trace on the boundary of K. Defaults to the zero
        Polynomial.
    has_poly_trace : bool
        Whether or not the local function has a piecewise Polynomial trace.
    lap : Polynomial
        Laplacian Polynomial.
    poly_part : Polynomial
        Polynomial part of the local function.
    poly_part_trace : numpy.ndarray
        Dirichlet trace values of the Polynomial part.
    poly_part_wnd : numpy.ndarray
        Weighted normal derivative of the Polynomial part.
    conj_trace : numpy.ndarray
        Dirichlet trace values of the harmonic conjugate of the harmonic part.
    log_coef : numpy.ndarray
        Logarithmic coefficients of the harmonic part.
    harm_part_wnd : numpy.ndarray
        Weighted normal derivative of the harmonic part.
    antilap_trace : numpy.ndarray
        Dirichlet trace values of an anti-Laplacian of the harmonic part.
    antilap_wnd : numpy.ndarray
        Weighted normal derivative of an anti-Laplacian of the harmonic part.
    nyst : NystromSolver
        Nystrom nyst object, which contains the MeshCell K.
    int_vals : numpy.ndarray
        Interior values of the local function.
    int_grad1 : numpy.ndarray
        First component of the gradient of the local function.
    int_grad2 : numpy.ndarray
        Second component of the gradient of the local function.
    """

    key: GlobalKey
    trace: np.ndarray
    poly_trace: PiecewisePolynomial
    has_poly_trace: bool
    lap: Polynomial
    poly_part: Polynomial
    poly_part_trace: np.ndarray
    poly_part_wnd: np.ndarray
    conj_trace: np.ndarray
    log_coef: np.ndarray
    harm_part_wnd: np.ndarray
    antilap_trace: np.ndarray
    antilap_wnd: np.ndarray
    nyst: NystromSolver
    int_vals: np.ndarray
    int_grad1: np.ndarray
    int_grad2: np.ndarray

    def __init__(
        self,
        nyst: NystromSolver,
        lap_poly: Polynomial = Polynomial(),
        poly_trace: Optional[PiecewisePolynomial] = None,
        has_poly_trace: bool = True,
        key: Optional[GlobalKey] = None,
    ) -> None:
        """
        Build an element of the local Poisson space V_p(K).

        Parameters
        ----------
        nyst : NystromSolver
            Nystrom nyst object, which contains the MeshCell K.
        lap_poly : Polynomial, optional
            Laplacian Polynomial, by default Polynomial()
        poly_trace : Optional[PiecewisePolynomial], optional
            Piecewise Polynomial trace, by default None
        has_poly_trace : bool, optional
            Whether or not the local function has a piecewise Polynomial trace,
            by default True
        key : Optional[GlobalKey], optional
            Global key, by default None
        """
        self.set_key(key)
        self.set_nystrom_solver(nyst)
        self.set_laplacian_polynomial(lap_poly)
        self.set_poly_trace(poly_trace)
        self.has_poly_trace = has_poly_trace

    def set_key(self, key: Optional[GlobalKey]) -> None:
        """
        Set the global key for the local function.

        Parameters
        ----------
        key : Optional[GlobalKey]
            Global key.
        """
        if key is None:
            return
        if not isinstance(key, GlobalKey):
            raise TypeError("key must be a GlobalKey")
        self.key = key

    def set_nystrom_solver(self, nyst: NystromSolver) -> None:
        """
        Set the Nystrom nyst for the local function.

        Parameters
        ----------
        nyst : NystromSolver
            Nystrom nyst object.
        """
        if not isinstance(nyst, NystromSolver):
            raise TypeError("nyst must be a NystromSolver")
        self.nyst = nyst

    def compute_all(self, compute_int_vals: bool = False) -> None:
        """
        Compute all quantities associated with the local function.

        The following quantities are computed:
            - Dirichlet trace values (if has_poly_trace is True)
            - Polynomial part (anti-Laplacian of polynomial Laplacian)
            - Polynomial part trace
            - Polynomial part weighted normal derivative
            - Harmonic conjugate of harmonic part
            - Weighted normal derivative of harmonic part
            - Anti-Laplacian of harmonic part
            - Interior values (if compute_int_vals is True)

        Parameters
        ----------
        compute_int_vals : bool, optional
            Whether or not to compute interior values, by default False.
        """
        if self.has_poly_trace:
            self.compute_trace_values()
        self.compute_polynomial_part()
        self.compute_polynomial_part_trace()
        self.compute_polynomial_part_weighted_normal_derivative()
        self.compute_harmonic_conjugate()
        self.compute_harmonic_weighted_normal_derivative()
        self.compute_anti_laplacian_harmonic_part()
        if compute_int_vals:
            self.compute_interior_values()

    def clear(self) -> None:
        """Delete all large np.ndarrays to save memory."""
        self.trace = np.zeros((0,))
        self.poly_part_trace = np.zeros((0,))
        self.poly_part_wnd = np.zeros((0,))
        self.conj_trace = np.zeros((0,))
        self.log_coef = np.zeros((0,))
        self.harm_part_wnd = np.zeros((0,))
        self.antilap_trace = np.zeros((0,))
        self.antilap_wnd = np.zeros((0,))

    # Piecewise Polynomial Dirichlet trace ###################################
    def set_poly_trace(self, poly_trace: Optional[PiecewisePolynomial]) -> None:
        """
        Set the piecewise Polynomial trace to self.poly_trace.

        Parameters
        ----------
        poly_trace : Optional[PiecewisePolynomial]
            Piecewise Polynomial trace.
        """
        if poly_trace is None:
            self.poly_trace = PiecewisePolynomial(
                num_polys=self.nyst.K.num_edges
            )
            return
        if not isinstance(poly_trace, PiecewisePolynomial):
            raise TypeError("poly_trace must be a PiecewisePolynomial")
        if poly_trace.num_polys != self.nyst.K.num_edges:
            raise ValueError(
                "Number of Polynomials must match number of edges in K"
            )
        self.poly_trace = poly_trace

    def get_poly_trace(self) -> PiecewisePolynomial:
        """
        Get the piecewise Polynomial trace.

        Returns
        -------
        PiecewisePolynomial
            Piecewise Polynomial trace.
        """
        return self.poly_trace

    # Dirichlet trace values #################################################
    def set_trace_values(self, vals: np.ndarray) -> None:
        """
        Set the Dirichlet trace values to self.trace.

        Parameters
        ----------
        vals : np.ndarray
            Dirichlet trace values.
        """
        self.trace = vals

    def get_trace_values(self) -> np.ndarray:
        """
        Get the Dirichlet trace values.

        Returns
        -------
        np.ndarray
            Dirichlet trace values.
        """
        return self.trace

    def compute_trace_values(self) -> None:
        """
        Compute the Dirichlet trace values.

        The trace values are stored in self.trace. The trace values are
        computed by evaluating the piecewise polynomial trace on the boundary,
        if it has been set.

        Raises
        ------
        SizeMismatchError
            If the local function does not have a piecewise Polynomial trace.
        """
        if not self.has_poly_trace:
            raise SizeMismatchError(
                "Cannot compute trace values for local function "
                + "without piecewise Polynomial trace"
            )
        if self.poly_trace.num_polys != self.nyst.K.num_edges:
            raise SizeMismatchError(
                "There must be exactly one trace Polynomial "
                + "for every Edge of K"
            )
        self.trace = np.zeros((self.nyst.K.num_pts,))
        edge_idx = 0
        for i in range(self.nyst.K.num_holes + 1):
            c = self.nyst.K.components[i]
            for j in range(c.num_edges):
                k_start = self.nyst.K.component_start_idx[i] + c.vert_idx[j]
                k_end = self.nyst.K.component_start_idx[i] + c.vert_idx[j + 1]
                self.trace[k_start:k_end] = self.poly_trace.polys[edge_idx](
                    c.edges[j].x[0, :-1], c.edges[j].x[1, :-1]
                )
                edge_idx += 1

    # Laplacian (Polynomial) #################################################
    def set_laplacian_polynomial(self, p: Polynomial) -> None:
        """
        Set the Laplacian Polynomial to self.lap.

        Parameters
        ----------
        p : Polynomial
            Polynomial representing the Laplacian: p = Delta v.

        Raises
        ------
        TypeError
            If p is not a Polynomial.
        """
        if not isinstance(p, Polynomial):
            raise TypeError("p must be a Polynomial")
        self.lap = p

    def get_laplacian_polynomial(self) -> Polynomial:
        """
        Get the Laplacian polynomial: p = Delta v.

        Returns
        -------
        Polynomial
            Polynomial representing the Laplacian: p = Delta v.
        """
        return self.lap

    # Polynomial part (Polynomial anti-Laplacian of Laplacian) ###############
    def set_polynomial_part(self, P_poly: Polynomial) -> None:
        """
        Set the polynomial part: P = v - phi.

        The polynomial part is stored in self.poly_part.

        Parameters
        ----------
        P_poly : Polynomial
            Polynomial part: P = v - phi.
        """
        if not isinstance(P_poly, Polynomial):
            raise TypeError("P_poly must be a Polynomial")
        self.poly_part = P_poly

    def get_polynomial_part(self) -> Polynomial:
        """
        Get the Polynomial part: P = v - phi.

        Returns
        -------
        Polynomial
            Polynomial part P = v - phi.
        """
        return self.poly_part

    def compute_polynomial_part(self) -> None:
        """
        Compute the polynomial part: P = v - phi.

        The Polynomial part is stored in self.poly_part.
        """
        self.poly_part = self.lap.anti_laplacian()

    # Polynomial part trace ##################################################
    def set_polynomial_part_trace(self, P_trace: np.ndarray) -> None:
        """
        Set the trace values of the polynomial part: P = v - phi.

        The trace values are stored in self.poly_part_trace.

        Parameters
        ----------
        P_trace : np.ndarray
            Dirichlet trace values of the Polynomial part: P = v - phi.
        """
        self.poly_part_trace = P_trace

    def get_polynomial_part_trace(self) -> np.ndarray:
        """
        Get the trace values of the Polynomial part: P = v - phi.

        Returns
        -------
        np.ndarray
            Dirichlet trace values of the Polynomial part.
        """
        return self.poly_part_trace

    def compute_polynomial_part_trace(self) -> None:
        """
        Compute the trace values of the polynomial part: P = v - phi.

        The trace values are stored in self.poly_part_trace.
        """
        x1, x2 = self.nyst.K.get_boundary_points()
        self.poly_part_trace = self.poly_part(x1, x2)

    # Polynomial part weighted normal derivative #############################
    def set_polynomial_part_weighted_normal_derivative(
        self, P_wnd: np.ndarray
    ) -> None:
        """
        Set the weighted normal derivative of the polynomial part: P = v - phi.

        The weighted normal derivative is stored in self.poly_part_wnd.

        Parameters
        ----------
        P_wnd : np.ndarray
            Weighted normal derivative of the Polynomial part: P = v - phi.
        """
        self.poly_part_wnd = P_wnd

    def get_polynomial_part_weighted_normal_derivative(self) -> np.ndarray:
        """
        Get the weighted normal derivative of the polynomial part.

        Returns
        -------
        np.ndarray
            Values of the weighted normal derivative of the polynomial part.
        """
        return self.poly_part_wnd

    def compute_polynomial_part_weighted_normal_derivative(self) -> None:
        """
        Compute the weighted normal derivative of the polynomial part.

        The weighted normal derivative is stored in self.poly_part_wnd.
        """
        x1, x2 = self.nyst.K.get_boundary_points()
        g1, g2 = self.poly_part.grad()
        P_nd = self.nyst.K.dot_with_normal(g1(x1, x2), g2(x1, x2))
        self.poly_part_wnd = self.nyst.K.multiply_by_dx_norm(P_nd)

    # harmonic part ##########################################################
    def get_harmonic_part_trace(self) -> np.ndarray:
        """
        Get the Dirichlet trace of the harmonic part: phi = v - P.

        Returns
        -------
        np.ndarray
            Dirichlet trace values of the harmonic part.
        """
        return self.trace - self.poly_part_trace

    # harmonic conjugate #####################################################
    def set_harmonic_conjugate(self, hc_vals: np.ndarray) -> None:
        """
        Set the trace values of the harmonic conjugate.

        Parameters
        ----------
        hc_vals : np.ndarray
            Dirichlet trace values of the harmonic conjugate.
        """
        self.conj_trace = hc_vals

    def get_harmonic_conjugate(self) -> np.ndarray:
        """
        Get the Dirichlet trace values of the harmonic conjugate.

        Returns
        -------
        np.ndarray
            Dirichlet trace values of the harmonic conjugate.
        """
        return self.conj_trace

    def compute_harmonic_conjugate(self) -> None:
        """
        Compute the Dirichlet trace values of the harmonic conjugate.

        The trace values are stored in self.conj_trace.
        """
        phi_trace = self.get_harmonic_part_trace()
        self.conj_trace, self.log_coef = self.nyst.get_harmonic_conjugate(
            phi_trace
        )

    # logarithmic coefficients ###############################################
    def set_logarithmic_coefficients(self, log_coef: np.ndarray) -> None:
        """
        Set the logarithmic coefficients of the harmonic part.

        The logarithmic coefficients are stored in self.log_coef.

        Parameters
        ----------
        log_coef : np.ndarray
            Logarithmic coefficients of the harmonic part.
        """
        self.log_coef = log_coef

    def get_logarithmic_coefficients(self) -> np.ndarray:
        """
        Get the logarithmic coefficients of the harmonic part.

        Returns
        -------
        np.ndarray
            Logarithmic coefficients of the harmonic part.
        """
        return self.log_coef

    # no compute method, this is handled by compute_harmonic_conjugate()

    # weighted normal derivative of harmonic part ############################
    def set_harmonic_weighted_normal_derivative(
        self, hc_wnd: np.ndarray
    ) -> None:
        """
        Set the weighted normal derivative of the harmonic part.

        The weighted normal derivative is stored in self.harm_part_wnd.

        Parameters
        ----------
        hc_wnd : np.ndarray
            Weighted normal derivative of the harmonic part.
        """
        self.harm_part_wnd = hc_wnd

    def get_harmonic_weighted_normal_derivative(self) -> np.ndarray:
        """
        Get the weighted normal derivative of the harmonic part.

        Returns
        -------
        np.ndarray
            Values of the weighted normal derivative of the harmonic part.
        """
        return self.harm_part_wnd

    def compute_harmonic_weighted_normal_derivative(self) -> None:
        """
        Compute the weighted normal derivative of the harmonic part.

        The weighted normal derivative is stored in self.harm_part_wnd.
        """
        self.harm_part_wnd = (
            d2n.trace2tangential.get_weighted_tangential_derivative_from_trace(
                self.nyst.K, self.conj_trace
            )
        )
        lam_x1, lam_x2 = d2n.log_terms.get_log_grad(self.nyst.K)
        lam_wnd = d2n.log_terms.get_dlam_dn_wgt(self.nyst.K, lam_x1, lam_x2)
        self.harm_part_wnd += lam_wnd @ self.log_coef

    # harmonic conjugable part psi ###########################################
    def get_conjugable_part(self) -> np.ndarray:
        """
        Return the harmonic conjugable part psi.

        The conjugable part of a harmonic function phi is the harmonic function
        psi such that phi = psi + sum_{j=1}^m a_j log |x - xi_j|, with psi
        having a harmonic conjugate psi_hat.

        Returns
        -------
        np.ndarray
            Dirichlet trace values of the harmonic conjugable part.
        """
        lam = d2n.log_terms.get_log_trace(self.nyst.K)
        phi = self.get_harmonic_part_trace()
        return phi - lam @ self.log_coef

    # anti-Laplacian #########################################################
    def set_anti_laplacian_harmonic_part(
        self, anti_laplacian_vals: np.ndarray
    ) -> None:
        """
        Set the trace values of an anti-Laplacian of the harmonic part.

        Parameters
        ----------
        anti_laplacian_vals : np.ndarray
            Dirichlet trace values of an anti-Laplacian of the harmonic part.
        """
        self.antilap_trace = anti_laplacian_vals

    def get_anti_laplacian_harmonic_part(self) -> np.ndarray:
        """
        Get the trace values of an anti-Laplacian of the harmonic part.

        Returns
        -------
        np.ndarray
            Dirichlet trace values of an anti-Laplacian of the harmonic part.
        """
        return self.antilap_trace

    def compute_anti_laplacian_harmonic_part(self) -> None:
        """
        Compute the trace values of an anti-Laplacian of the harmonic part.

        The trace values are stored in self.antilap_trace.
        """
        psi = self.get_conjugable_part()
        psi_hat = self.conj_trace
        (
            self.antilap_trace,
            self.antilap_wnd,
        ) = antilap.antilap.get_anti_laplacian_harmonic(
            self.nyst, psi, psi_hat, a=self.log_coef
        )

    # H^1 semi-inner product #################################################
    def get_h1_semi_inner_prod(self, other: object) -> float:
        """
        Return the H^1 semi-inner product int_K grad(self) * grad(other) dx.

        Parameters
        ----------
        other : object
            Another LocalFunction.
        """
        if not isinstance(other, LocalFunction):
            raise TypeError("other must be aLocalFunction")

        # Polynomial part
        Px, Py = self.poly_part.grad()
        Qx, Qy = other.poly_part.grad()
        gradP_gradQ = Px * Qx + Py * Qy
        val = integrate_poly_over_mesh(gradP_gradQ, self.nyst.K)

        # remaining terms
        integrand = (
            other.trace * self.harm_part_wnd
            + self.poly_part_trace * other.harm_part_wnd
        )
        val += self.nyst.K.integrate_over_boundary_preweighted(integrand)

        return val

    # L^2 inner product ######################################################
    def get_l2_inner_prod(self, other: object) -> float:
        """
        Return the L^2 inner product int_K (self) * (other) dx.

        Parameters
        ----------
        other : object
            Another LocalFunction.
        """
        if not isinstance(other, LocalFunction):
            raise TypeError("other must be aLocalFunction")

        x1, x2 = self.nyst.K.get_boundary_points()

        # P * Q
        PQ = self.poly_part * other.poly_part
        val = integrate_poly_over_mesh(PQ, self.nyst.K)

        # phi * psi
        integrand = (
            other.trace - other.poly_part_trace
        ) * self.antilap_wnd - self.antilap_trace * other.harm_part_wnd

        # phi * Q
        R = other.poly_part.anti_laplacian()
        R_trace = R(x1, x2)
        R_wnd = R.get_weighted_normal_derivative(self.nyst.K)
        integrand += (
            self.trace - self.poly_part_trace
        ) * R_wnd - R_trace * self.harm_part_wnd

        # psi * P
        R = self.poly_part.anti_laplacian()
        R_trace = R(x1, x2)
        R_wnd = R.get_weighted_normal_derivative(self.nyst.K)
        integrand += (
            other.trace - other.poly_part_trace
        ) * R_wnd - R_trace * other.harm_part_wnd

        # integrate over boundary
        val += self.nyst.K.integrate_over_boundary_preweighted(integrand)

        return val

    # INTERIOR VALUES ########################################################

    def compute_interior_values(self, compute_grad: bool = True) -> None:
        """
        Compute the interior values.

        Also compute the components of the gradient if compute_grad is True. The
        interior values are stored in self.int_vals, and the gradient components
        are stored in self.int_grad1 and self.int_grad2.

        Parameters
        ----------
        compute_grad : bool, optional
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
        if compute_grad:
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
            if compute_grad:
                grad1 += self.log_coef[k] * y_xi_1 / y_xi_norm_sq
                grad2 += self.log_coef[k] * y_xi_2 / y_xi_norm_sq

        # conjugable part
        psi = self.get_conjugable_part()
        psi_hat = self.get_harmonic_conjugate()

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
        if compute_grad:
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
        if compute_grad:
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
