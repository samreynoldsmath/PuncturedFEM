"""
locfun.py
=========

Module containing the locfun class for managing elements of the local
Poisson space V_p(K).
"""

from typing import Optional

from numpy import empty, log, nan, ndarray, pi, zeros

from ..mesh.mesh_exceptions import SizeMismatchError
from ..solver.globkey import global_key
from . import antilap, d2n
from .nystrom import nystrom_solver
from .poly.integrate_poly import integrate_poly_over_cell
from .poly.piecewise_poly import piecewise_polynomial
from .poly.poly import polynomial


class locfun:
    """
    Element of the local Poisson space V_p(K), whose trace is continuous and a
    polynomial of degree <= p on each edge of K, and whose Laplacian is a
    polynomial of degree <= p-2 on each edge of K.

    Can also be used to represent functions with an arbitrary continuous trace
    on the boundary of K.

    Any such function v can be decomposed as
        v = P + phi
    where P is a polynomial of degree <= p in K and phi is a harmonic function.
    (Note that this decomposition is not unique, since there are harmonic
    polynomials). We refer to P as the "polynomial part" and phi as the
    "harmonic part".

    Furthermore, in multiply connected cells the harmonic part can be
    decomposed as
        phi = psi + sum_{j=1}^m a_j log |x - xi_j|
    where xi_j is a fixed arbitrary point in the jth hole of K, and psi is a
    harmonic function with a harmonic conjugate psi_hat. We refer to psi as the
    "conjugable part".

    Given a parametrization x(tau) of the boundary of K, we refer to
        (d / dtau) v(x(tau)) = nabla v(x(tau)) * x'(tau)
    as the "weighted normal derivative" of v, which we commonly abbreviate in
    code with "wnd". Similarly, the "weighted tangential derivative" is
    abbreviated "wtd".

    An "anti-Laplacian" of u is any function U such that
        Delta U = u
    where Delta is the Laplacian operator. In particular, we desire an anti-
    Laplacian of the harmonic part phi, as well as its weighted normal
    derivative. This allows us to compute the L^2 inner product of two local
    functions v and w using only boundary integrals.

    Interior values of a local function v, as well as its gradient, can be
    computed using Cauchy's integral formula.


    Usage
    -----
    Define an instance by passing a Nystrom solver object, a its Laplacian
    polynomial, and (optionally) a piecewise polynomial trace:
        >>> v = locfun(solver, lap_poly, poly_trace)
    where poly_trace defaults to the zero polynomial if not specified. If one
    wishes to use a local function with an arbitrary continuous trace, then
    the has_poly_trace flag should be set to False:
        >>> v = locfun(solver, lap_poly, has_poly_trace=False)
    The trace values can be set using the set_trace_values method:
        >>> v.set_trace_values(trace_vals)
    Executing the compute_all method will compute and store all relevant data
    needed to compute volumetric integrals:
        >>> v.compute_all()
    Given another local function w, the H^1 semi-inner product can be computed
    using the get_h1_semi_inner_prod method:
        >>> val = v.get_h1_semi_inner_prod(w)
    Similarly, the L^2 inner product can be computed using the
    get_l2_inner_prod method:
        >>> val = v.get_l2_inner_prod(w)
    Interior values of v can be computed using the compute_interior_values
    method:
        >>> v.compute_interior_values()
    and retrieved using the get_interior_values method:
        >>> vals = v.get_interior_values()
    The interior values are stored in v.int_vals, with the same shape and
    indexing as the interior points of the cell v.solver.K. The components of
    the gradient are stored in v.int_grad1 and v.int_grad2, respectively.
    The "punctured square" example
        examples/ex1a-square-hole.ipynb
    demonstrates more detailed usage of the locfun class.

    Attributes
    ----------
    key : global_key
        A unique tag that identifies the local function in the global space.
    trace : ndarray
        Dirichlet trace values on the boundary of K.
    poly_trace : piecewise_polynomial
        Piecewise polynomial trace on the boundary of K. Defaults to the zero
        polynomial.
    has_poly_trace : bool
        Whether or not the local function has a piecewise polynomial trace.
    lap : polynomial
        Laplacian polynomial.
    poly_part : polynomial
        Polynomial part of the local function.
    poly_part_trace : ndarray
        Dirichlet trace values of the polynomial part.
    poly_part_wnd : ndarray
        Weighted normal derivative of the polynomial part.
    conj_trace : ndarray
        Dirichlet trace values of the harmonic conjugate of the harmonic part.
    log_coef : ndarray
        Logarithmic coefficients of the harmonic part.
    harm_part_wnd : ndarray
        Weighted normal derivative of the harmonic part.
    antilap_trace : ndarray
        Dirichlet trace values of an anti-Laplacian of the harmonic part.
    antilap_wnd : ndarray
        Weighted normal derivative of an anti-Laplacian of the harmonic part.
    solver : nystrom_solver
        Nystrom solver object, which contains the cell K.
    int_vals : ndarray
        Interior values of the local function.
    int_grad1 : ndarray
        First component of the gradient of the local function.
    int_grad2 : ndarray
        Second component of the gradient of the local function.
    """

    key: global_key
    trace: ndarray
    poly_trace: piecewise_polynomial
    has_poly_trace: bool
    lap: polynomial
    poly_part: polynomial
    poly_part_trace: ndarray
    poly_part_wnd: ndarray
    conj_trace: ndarray
    log_coef: ndarray
    harm_part_wnd: ndarray
    antilap_trace: ndarray
    antilap_wnd: ndarray
    solver: nystrom_solver
    int_vals: ndarray
    int_grad1: ndarray
    int_grad2: ndarray

    def __init__(
        self,
        solver: nystrom_solver,
        lap_poly: polynomial = polynomial(),  # TODO maybe should be None?
        poly_trace: Optional[piecewise_polynomial] = None,
        has_poly_trace: bool = True,
        key: Optional[global_key] = None,
    ) -> None:
        """
        Constructor for locfun class.

        Parameters
        ----------
        solver : nystrom_solver
            Nystrom solver object, which contains the cell K.
        lap_poly : polynomial, optional
            Laplacian polynomial, by default polynomial()
        poly_trace : Optional[piecewise_polynomial], optional
            Piecewise polynomial trace, by default None
        has_poly_trace : bool, optional
            Whether or not the local function has a piecewise polynomial trace,
            by default True
        id : Optional[global_key], optional
            Global key, by default None
        """
        self.set_key(key)
        self.set_solver(solver)
        self.set_laplacian_polynomial(lap_poly)
        self.set_poly_trace(poly_trace)
        self.has_poly_trace = has_poly_trace

    def set_key(self, key: Optional[global_key]) -> None:
        """
        Sets the global key for the local function.
        """
        if key is None:
            return
        if not isinstance(key, global_key):
            raise TypeError("key must be a global_key")
        self.key = key

    def set_solver(self, solver: nystrom_solver) -> None:
        """
        Sets the Nystrom solver for the local function.
        """
        if not isinstance(solver, nystrom_solver):
            raise TypeError("solver must be a nystrom_solver")
        self.solver = solver

    def compute_all(self) -> None:
        """
        Computes all relevant data for reducing volumetric integrals
        to boundary integrals
        """
        if self.has_poly_trace:
            self.compute_trace_values()
        self.compute_polynomial_part()
        self.compute_polynomial_part_trace()
        self.compute_polynomial_part_weighted_normal_derivative()
        self.compute_harmonic_conjugate()
        self.compute_harmonic_weighted_normal_derivative()
        self.compute_anti_laplacian_harmonic_part()

    def clear(self) -> None:
        """
        Deletes all large ndarrays (to save memory)
        """
        self.trace = zeros((0,))
        self.poly_part_trace = zeros((0,))
        self.poly_part_wnd = zeros((0,))
        self.conj_trace = zeros((0,))
        self.log_coef = zeros((0,))
        self.harm_part_wnd = zeros((0,))
        self.antilap_trace = zeros((0,))
        self.antilap_wnd = zeros((0,))

    # Piecewise polynomial Dirichlet trace ###################################
    def set_poly_trace(
        self, poly_trace: Optional[piecewise_polynomial]
    ) -> None:
        """
        Sets the piecewise polynomial trace to self.poly_trace.
        """
        if poly_trace is None:
            self.poly_trace = piecewise_polynomial(
                num_polys=self.solver.K.num_edges
            )
            return
        if not isinstance(poly_trace, piecewise_polynomial):
            raise TypeError("poly_trace must be a piecewise_polynomial")
        if poly_trace.num_polys != self.solver.K.num_edges:
            raise ValueError(
                "Number of polynomials must match number of edges in K"
            )
        self.poly_trace = poly_trace

    def get_poly_trace(self) -> piecewise_polynomial:
        """
        Returns the piecewise polynomial trace.
        """
        return self.poly_trace

    # Dirichlet trace values #################################################
    def set_trace_values(self, vals: ndarray) -> None:
        """
        Sets the Dirichlet trace values to self.trace.
        """
        self.trace = vals

    def get_trace_values(self) -> ndarray:
        """
        Returns the Dirichlet trace values.
        """
        return self.trace

    def compute_trace_values(self) -> None:
        """
        Computes the Dirichlet trace values from the piecewise polynomial trace
        and stores them in self.trace.
        """
        if not self.has_poly_trace:
            raise SizeMismatchError(
                "Cannot compute trace values for local function "
                + "without piecewise polynomial trace"
            )
        if self.poly_trace.num_polys != self.solver.K.num_edges:
            raise SizeMismatchError(
                "There must be exactly one trace polynomial "
                + "for every edge of K"
            )
        self.trace = zeros((self.solver.K.num_pts,))
        edge_idx = 0
        for i in range(self.solver.K.num_holes + 1):
            c = self.solver.K.components[i]
            for j in range(c.num_edges):
                k_start = self.solver.K.component_start_idx[i] + c.vert_idx[j]
                k_end = self.solver.K.component_start_idx[i] + c.vert_idx[j + 1]
                self.trace[k_start:k_end] = self.poly_trace.polys[
                    edge_idx
                ].eval(c.edges[j].x[0, :-1], c.edges[j].x[1, :-1])
                edge_idx += 1

    # Laplacian (polynomial) #################################################
    def set_laplacian_polynomial(self, p: polynomial) -> None:
        """
        Sets the Laplacian polynomial to self.lap.
        """
        if not isinstance(p, polynomial):
            raise TypeError("p must be a polynomial")
        self.lap = p

    def get_laplacian_polynomial(self) -> polynomial:
        """
        Returns the Laplacian polynomial.
        """
        return self.lap

    # polynomial part (polynomial anti-Laplacian of Laplacian) ###############
    def set_polynomial_part(self, P_poly: polynomial) -> None:
        """
        Sets the polynomial part to self.poly_part.
        """
        if not isinstance(P_poly, polynomial):
            raise TypeError("P_poly must be a polynomial")
        self.poly_part = P_poly

    def get_polynomial_part(self) -> polynomial:
        """
        Returns the polynomial part.
        """
        return self.poly_part

    def compute_polynomial_part(self) -> None:
        """
        Computes the polynomial part from the Laplacian polynomial and stores
        it in self.poly_part.
        """
        self.poly_part = self.lap.anti_laplacian()

    # polynomial part trace ##################################################
    def set_polynomial_part_trace(self, P_trace: ndarray) -> None:
        """
        Sets the Dirichlet trace values of the polynomial part to
        self.poly_part_trace.
        """
        self.poly_part_trace = P_trace

    def get_polynomial_part_trace(self) -> ndarray:
        """
        Returns the Dirichlet trace values of the polynomial part.
        """
        return self.poly_part_trace

    def compute_polynomial_part_trace(self) -> None:
        """
        Computes the Dirichlet trace values of the polynomial part and stores
        them in self.poly_part_trace.
        """
        x1, x2 = self.solver.K.get_boundary_points()
        self.poly_part_trace = self.poly_part.eval(x1, x2)

    # polynomial part weighted normal derivative #############################
    def set_polynomial_part_weighted_normal_derivative(
        self, P_wnd: ndarray
    ) -> None:
        """
        Sets the weighted normal derivative of the polynomial part to
        self.poly_part_wnd.
        """
        self.poly_part_wnd = P_wnd

    def get_polynomial_part_weighted_normal_derivative(self) -> ndarray:
        """
        Returns the weighted normal derivative of the polynomial part.
        """
        return self.poly_part_wnd

    def compute_polynomial_part_weighted_normal_derivative(self) -> None:
        """
        Computes the weighted normal derivative of the polynomial part and
        stores it in self.poly_part_wnd.
        """
        x1, x2 = self.solver.K.get_boundary_points()
        g1, g2 = self.poly_part.grad()
        P_nd = self.solver.K.dot_with_normal(g1.eval(x1, x2), g2.eval(x1, x2))
        self.poly_part_wnd = self.solver.K.multiply_by_dx_norm(P_nd)

    # harmonic conjugate #####################################################
    def set_harmonic_conjugate(self, hc_vals: ndarray) -> None:
        """
        Sets the Dirichlet trace values of the harmonic conjugate of the
        harmonic part to self.conj_trace.
        """
        self.conj_trace = hc_vals

    def get_harmonic_conjugate(self) -> ndarray:
        """
        Returns the Dirichlet trace values of the harmonic conjugate of the
        harmonic part.
        """
        return self.conj_trace

    def compute_harmonic_conjugate(self) -> None:
        """
        Computes the Dirichlet trace values of the harmonic conjugate of the
        harmonic part and stores them in self.conj_trace.
        """
        phi_trace = self.trace - self.poly_part_trace
        self.conj_trace, self.log_coef = self.solver.get_harmonic_conjugate(
            phi_trace
        )

    # logarithmic coefficients ###############################################
    def set_logarithmic_coefficients(self, log_coef: ndarray) -> None:
        """
        Sets the logarithmic coefficients of the harmonic part to self.log_coef.
        """
        self.log_coef = log_coef

    def get_logarithmic_coefficients(self) -> ndarray:
        """
        Returns the logarithmic coefficients of the harmonic part.
        """
        return self.log_coef

    # no compute method, this is handled by compute_harmonic_conjugate()

    # weighted normal derivative of harmonic part ############################
    def set_harmonic_weighted_normal_derivative(self, hc_wnd: ndarray) -> None:
        """
        Sets the weighted normal derivative of the harmonic part to
        self.harm_part_wnd.
        """
        self.harm_part_wnd = hc_wnd

    def get_harmonic_weighted_normal_derivative(self) -> ndarray:
        """
        Returns the weighted normal derivative of the harmonic part.
        """
        return self.harm_part_wnd

    def compute_harmonic_weighted_normal_derivative(self) -> None:
        """
        Computes the weighted normal derivative of the harmonic part and stores
        it in self.harm_part_wnd.
        """
        self.harm_part_wnd = (
            d2n.trace2tangential.get_weighted_tangential_derivative_from_trace(
                self.solver.K, self.conj_trace
            )
        )
        lam_x1, lam_x2 = d2n.log_terms.get_log_grad(self.solver.K)
        lam_wnd = d2n.log_terms.get_dlam_dn_wgt(self.solver.K, lam_x1, lam_x2)
        self.harm_part_wnd += lam_wnd @ self.log_coef

    # harmonic conjugable part psi ###########################################
    def get_conjugable_part(self) -> ndarray:
        """
        Returns the harmonic conjugable part psi.
        """
        lam = d2n.log_terms.get_log_trace(self.solver.K)
        return self.trace - self.poly_part_trace - lam @ self.log_coef

    # anti-Laplacian #########################################################
    def set_anti_laplacian_harmonic_part(
        self, anti_laplacian_vals: ndarray
    ) -> None:
        """
        Sets the Dirichlet trace values of an anti-Laplacian of the harmonic
        part to self.antilap_trace.
        """
        self.antilap_trace = anti_laplacian_vals

    def get_anti_laplacian_harmonic_part(self) -> ndarray:
        """
        Returns the Dirichlet trace values of an anti-Laplacian of the harmonic
        part.
        """
        return self.antilap_trace

    def compute_anti_laplacian_harmonic_part(self) -> None:
        """
        Computes the Dirichlet trace values of an anti-Laplacian of the
        harmonic part and stores them in self.antilap_trace.
        """
        psi = self.get_conjugable_part()
        (
            self.antilap_trace,
            self.antilap_wnd,
        ) = antilap.antilap.get_anti_laplacian_harmonic(
            self.solver.K, psi=psi, psi_hat=self.conj_trace, a=self.log_coef
        )

    # H^1 semi-inner product #################################################
    def get_h1_semi_inner_prod(self, other: object) -> float:
        """
        Returns the H^1 semi-inner product
            int_K grad(self) * grad(other) dx
        """

        if not isinstance(other, locfun):
            raise TypeError("other must be a locfun")

        # polynomial part
        Px, Py = self.poly_part.grad()
        Qx, Qy = other.poly_part.grad()
        gradP_gradQ = Px * Qx + Py * Qy
        val = integrate_poly_over_cell(gradP_gradQ, self.solver.K)

        # remaining terms
        integrand = (
            other.trace * self.harm_part_wnd
            + self.poly_part_trace * other.harm_part_wnd
        )
        val += self.solver.K.integrate_over_boundary_preweighted(integrand)

        return val

    # L^2 inner product ######################################################
    def get_l2_inner_prod(self, other: object) -> float:
        """
        Returns the L^2 inner product
            int_K (self) * (other) dx
        """

        if not isinstance(other, locfun):
            raise TypeError("other must be a locfun")

        x1, x2 = self.solver.K.get_boundary_points()

        # P * Q
        PQ = self.poly_part * other.poly_part
        val = integrate_poly_over_cell(PQ, self.solver.K)

        # phi * psi
        integrand = (
            other.trace - other.poly_part_trace
        ) * self.antilap_wnd - self.antilap_trace * other.harm_part_wnd

        # phi * Q
        R = other.poly_part.anti_laplacian()
        R_trace = R.eval(x1, x2)
        R_wnd = R.get_weighted_normal_derivative(self.solver.K)
        integrand += (
            self.trace - self.poly_part_trace
        ) * R_wnd - R_trace * self.harm_part_wnd

        # psi * P
        R = self.poly_part.anti_laplacian()
        R_trace = R.eval(x1, x2)
        R_wnd = R.get_weighted_normal_derivative(self.solver.K)
        integrand += (
            other.trace - other.poly_part_trace
        ) * R_wnd - R_trace * other.harm_part_wnd

        # integrate over boundary
        val += self.solver.K.integrate_over_boundary_preweighted(integrand)

        return val

    # INTERIOR VALUES ########################################################

    # TODO: move interior value calculation to separate module

    def compute_interior_values(self) -> None:
        """
        Computes the interior values and stores them in self.int_vals. Also
        computes the components of the gradient and stores them in
        self.int_grad1 and self.int_grad2.
        """

        # size of interior mesh
        rows, cols = self.solver.K.int_mesh_size

        # initialize arrays
        self.int_vals = empty((rows, cols))
        self.int_grad1 = empty((rows, cols))
        self.int_grad2 = empty((rows, cols))

        self.int_vals[:] = nan
        self.int_grad1[:] = nan
        self.int_grad2[:] = nan

        # points for evaluation
        int_x1 = self.solver.K.int_x1
        int_x2 = self.solver.K.int_x2

        # boundary points
        bdy_x1, bdy_x2 = self.solver.K.get_boundary_points()

        # conjugable part
        psi = self.get_conjugable_part()
        psi_hat = self.get_harmonic_conjugate()

        # polynomial gradient
        Px, Py = self.poly_part.grad()

        # compute interior values
        for i in range(rows):
            for j in range(cols):
                if self.solver.K.is_inside[i, j]:
                    self.cauchy_integral_formula(
                        i,
                        j,
                        bdy_x1,
                        bdy_x2,
                        int_x1,
                        int_x2,
                        psi,
                        psi_hat,
                        Px,
                        Py,
                    )

    def cauchy_integral_formula(
        self,
        i: int,
        j: int,
        bdy_x1: ndarray,
        bdy_x2: ndarray,
        int_x1: ndarray,
        int_x2: ndarray,
        psi: ndarray,
        psi_hat: ndarray,
        Px: polynomial,
        Py: polynomial,
    ) -> None:
        """
        Applies Cauchy's integral formula to compute the interior values and
        gradient components at the point (int_x1[i,j], int_x2[i,j]).
        """

        # Cauchy's integral formula
        xy1 = bdy_x1 - int_x1[i, j]
        xy2 = bdy_x2 - int_x2[i, j]
        xy_norm_sq = xy1 * xy1 + xy2 * xy2
        eta = (xy1 * psi + xy2 * psi_hat) / xy_norm_sq
        eta_hat = (xy1 * psi_hat - xy2 * psi) / xy_norm_sq
        integrand = self.solver.K.dot_with_tangent(eta_hat, eta)
        self.int_vals[i, j] = (
            self.solver.K.integrate_over_boundary(integrand) * 0.5 / pi
        )

        # polynomial part
        self.int_vals[i, j] += self.poly_part.eval(int_x1[i, j], int_x2[i, j])

        # logarithmic part
        for k in range(self.solver.K.num_holes):
            xi = self.solver.K.components[k + 1].interior_point
            y_xi_norm_sq = (int_x1[i, j] - xi.x) ** 2 + (
                int_x2[i, j] - xi.y
            ) ** 2
            self.int_vals[i, j] += 0.5 * self.log_coef[k] * log(y_xi_norm_sq)

        # Cauchy's integral formula for gradient
        omega = (xy1 * eta + xy2 * eta_hat) / xy_norm_sq
        omega_hat = (xy1 * eta_hat - xy2 * eta) / xy_norm_sq
        integrand = self.solver.K.dot_with_tangent(omega_hat, omega)
        self.int_grad1[i, j] = (
            self.solver.K.integrate_over_boundary(integrand) * 0.5 / pi
        )
        integrand = self.solver.K.dot_with_tangent(omega, -omega_hat)
        self.int_grad2[i, j] = (
            self.solver.K.integrate_over_boundary(integrand) * 0.5 / pi
        )

        # gradient polynomial part
        self.int_grad1[i, j] += Px.eval(int_x1[i, j], int_x2[i, j])
        self.int_grad2[i, j] += Py.eval(int_x1[i, j], int_x2[i, j])

        # gradient logarithmic part
        for k in range(self.solver.K.num_holes):
            xi = self.solver.K.components[k + 1].interior_point
            y_xi_1 = int_x1[i, j] - xi.x
            y_xi_2 = int_x2[i, j] - xi.y
            y_xi_norm_sq = y_xi_1**2 + y_xi_2**2
            self.int_grad1[i, j] += self.log_coef[k] * y_xi_1 / y_xi_norm_sq
            self.int_grad2[i, j] += self.log_coef[k] * y_xi_2 / y_xi_norm_sq
