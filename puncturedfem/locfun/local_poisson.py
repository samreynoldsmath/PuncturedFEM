"""
Elements of the local Poisson space V_p(K).

Classes
-------
LocalPoissonFunction
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..mesh.cell import MeshCell
from ..solver.globkey import GlobalKey
from ..util.types import FloatLike
from .inner_prod import h1_semi_inner_prod, l2_inner_prod
from .local_harmonic import LocalHarmonic
from .local_polynomial import LocalPolynomial
from .nystrom import NystromSolver
from .poly.poly import Polynomial
from .trace import DirichletTrace


class LocalPoissonFunction:
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
        (d / d tau) v(x(tau)) = nabla v(x(tau)) * x'(tau)
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
    harm : LocalHarmonic
        The harmonic part, phi.
    poly : LocalPolynomial
        The polynomial part, P.
    mesh_cell : MeshCell
        The mesh cell on which the local function is defined.
    int_vals : np.ndarray
        Interior values of the local function, evaluated on the interior mesh
        defined by the mesh cell.
    int_grad1 : np.ndarray
        First component of the gradient of the local function, evaluated on the
        interior mesh defined by the mesh cell.
    int_grad2 : np.ndarray
        Second component of the gradient of the local function, evaluated on the
        interior mesh defined by the mesh cell.
    key : GlobalKey
        A unique tag that identifies the local function in the global space.
    """

    harm: LocalHarmonic
    poly: LocalPolynomial
    mesh_cell: MeshCell
    int_vals: Optional[np.ndarray]
    int_grad1: Optional[np.ndarray]
    int_grad2: Optional[np.ndarray]
    key: GlobalKey

    def __init__(
        self,
        nyst: Optional[NystromSolver],
        laplacian: Polynomial = Polynomial(),
        trace: Union[DirichletTrace, FloatLike] = 0,
        evaluate_interior: bool = True,
        evaluate_gradient: bool = False,
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
        evaluate_interior : bool, optional
            Whether or not to compute the interior values, by default False.
        evaluate_gradient : bool, optional
            Whether or not to compute the gradient, by default False. Takes
            precedence over evaluate_interior.
        key : Optional[GlobalKey], optional
            A unique tag that identifies the local function in the global space.
        """
        self.int_vals = None
        self.int_grad1 = None
        self.int_grad2 = None
        if nyst is None:
            return
        self._set_key(key)
        self._set_mesh_cell(nyst.K)
        P = laplacian.anti_laplacian()
        self.poly = LocalPolynomial(P, nyst.K)
        if isinstance(trace, (float, int, np.ndarray)):
            trace = DirichletTrace(edges=nyst.K.get_edges(), values=trace)
        harm_trace_vals = DirichletTrace(
            edges=nyst.K.get_edges(),
            values=trace.values - self.poly.trace.values,
        )
        self.harm = LocalHarmonic(harm_trace_vals, nyst)
        if evaluate_interior or evaluate_gradient:
            self.compute_interior_values(evaluate_gradient)

    def __add__(self, other: LocalPoissonFunction) -> LocalPoissonFunction:
        """
        Add two local Poisson functions.

        Parameters
        ----------
        other : LocalPoissonFunction
            The other local Poisson function.

        Returns
        -------
        LocalPoissonFunction
            The sum of the two local Poisson functions.
        """
        if not isinstance(other, LocalPoissonFunction):
            raise TypeError("other must be a LocalPoissonFunction")
        new = LocalPoissonFunction(nyst=None)
        new._set_mesh_cell(self.mesh_cell)
        new.poly = self.poly + other.poly
        new.harm = self.harm + other.harm
        if self.int_vals is not None and other.int_vals is not None:
            new.int_vals = self.int_vals + other.int_vals
        if self.int_grad1 is not None and other.int_grad1 is not None:
            new.int_grad1 = self.int_grad1 + other.int_grad1
        if self.int_grad2 is not None and other.int_grad2 is not None:
            new.int_grad2 = self.int_grad2 + other.int_grad2
        return new

    def __mul__(self, other: Union[int, float]) -> LocalPoissonFunction:
        """
        Multiply the local Poisson function by a scalar.

        Parameters
        ----------
        other : Union[int, float]
            The scalar.

        Returns
        -------
        LocalPoissonFunction
            The product of the local Poisson function and the scalar.
        """
        if not isinstance(other, (int, float)):
            raise TypeError("other must be an int or a float")
        new = LocalPoissonFunction(nyst=None)
        new._set_mesh_cell(self.mesh_cell)
        new.poly = self.poly * other
        new.harm = self.harm * other
        if self.int_vals is not None:
            new.int_vals = self.int_vals * other
        if self.int_grad1 is not None:
            new.int_grad1 = self.int_grad1 * other
        if self.int_grad2 is not None:
            new.int_grad2 = self.int_grad2 * other
        return new

    def __rmul__(self, other: Union[int, float]) -> LocalPoissonFunction:
        """
        Multiply the local Poisson function by a scalar.

        Parameters
        ----------
        other : Union[int, float]
            The scalar.

        Returns
        -------
        LocalPoissonFunction
            The product of the local Poisson function and the scalar.
        """
        return self.__mul__(other)

    def __sub__(self, other: LocalPoissonFunction) -> LocalPoissonFunction:
        """
        Subtract two local Poisson functions.

        Parameters
        ----------
        other : LocalPoissonFunction
            The other local Poisson function.

        Returns
        -------
        LocalPoissonFunction
            The difference of the two local Poisson functions.
        """
        return self + other * -1

    def __truediv__(self, other: Union[int, float]) -> LocalPoissonFunction:
        """
        Divide the local Poisson function by a scalar.

        Parameters
        ----------
        other : Union[int, float]
            The scalar.

        Returns
        -------
        LocalPoissonFunction
            The division of the local Poisson function by the scalar.
        """
        if not isinstance(other, (int, float)):
            raise TypeError("other must be an int or a float")
        if other == 0:
            raise ValueError("Division by zero")
        return self * (1 / other)

    def get_trace_values(self) -> np.ndarray:
        """
        Return the trace values along the boundary of the mesh cell.

        Returns
        -------
        vals : numpy.ndarray
            Values of the trace, traverse in the same order as the sampled
            points on the boundary of the mesh cell.
        """
        return self.harm.trace.values + self.poly.trace.values

    def get_h1_semi_inner_prod(
        self, other: Union[LocalPoissonFunction, LocalHarmonic, LocalPolynomial]
    ) -> float:
        """
        Return the H^1 semi-inner product int_K grad(self) * grad(other) dx.

        Parameters
        ----------
        other : LocalPoissonFunction, LocalHarmonic, or LocalPolynomial
            Another local function on the same mesh cell.
        """
        if not isinstance(
            other, (LocalPoissonFunction, LocalHarmonic, LocalPolynomial)
        ):
            raise TypeError(
                "other must be a LocalPoissonFunction, LocalHarmonic, or "
                + "LocalPolynomial"
            )
        if isinstance(other, LocalPoissonFunction):
            val = h1_semi_inner_prod(self.harm, other.harm, self.mesh_cell)
            val += h1_semi_inner_prod(self.poly, other.poly, self.mesh_cell)
            val += h1_semi_inner_prod(self.harm, other.poly, self.mesh_cell)
            val += h1_semi_inner_prod(self.poly, other.harm, self.mesh_cell)
            return val
        # other is a LocalHarmonic or LocalPolynomial
        val = h1_semi_inner_prod(self.harm, other, self.mesh_cell)
        val += h1_semi_inner_prod(self.poly, other, self.mesh_cell)
        return val

    def get_l2_inner_prod(
        self, other: Union[LocalPoissonFunction, LocalHarmonic, LocalPolynomial]
    ) -> float:
        """
        Return the L^2 inner product int_K self * other dx.

        Parameters
        ----------
        other : LocalPoissonFunction, LocalHarmonic, or LocalPolynomial
            Another local function on the same mesh cell.
        """
        if not isinstance(
            other, (LocalPoissonFunction, LocalHarmonic, LocalPolynomial)
        ):
            raise TypeError(
                "other must be a LocalPoissonFunction, LocalHarmonic, or "
                + "LocalPolynomial"
            )
        if isinstance(other, LocalPoissonFunction):
            val = l2_inner_prod(self.harm, other.harm, self.mesh_cell)
            val += l2_inner_prod(self.harm, other.poly, self.mesh_cell)
            val += l2_inner_prod(self.poly, other.harm, self.mesh_cell)
            val += l2_inner_prod(self.poly, other.poly, self.mesh_cell)
            return val
        # other is a LocalHarmonic or LocalPolynomial
        val = l2_inner_prod(self.harm, other, self.mesh_cell)
        val += l2_inner_prod(self.poly, other, self.mesh_cell)
        return val

    def _set_key(self, key: Optional[GlobalKey]) -> None:
        if key is None:
            return
        if not isinstance(key, GlobalKey):
            raise TypeError("key must be a GlobalKey")
        self.key = key

    def _set_mesh_cell(self, mesh_cell: MeshCell) -> None:
        if not isinstance(mesh_cell, MeshCell):
            raise TypeError("mesh_cell must be a MeshCell")
        self.mesh_cell = mesh_cell

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
        y1 = self.mesh_cell.int_x1
        y2 = self.mesh_cell.int_x2

        # initialize temporary arrays
        N = len(y1)
        self.int_vals = np.zeros((N,))
        self.int_grad1 = np.zeros((N,))
        self.int_grad2 = np.zeros((N,))

        # polynomial part
        int_vals = self.poly.exact_form(y1, y2)
        if not isinstance(int_vals, np.ndarray):
            raise TypeError("The polynomial part must be a numpy array")
        self.int_vals += int_vals

        # gradient Polynomial part
        if compute_int_grad:
            int_grad1 = self.poly.grad1(y1, y2)
            int_grad2 = self.poly.grad2(y1, y2)
            if not isinstance(int_grad1, np.ndarray):
                raise TypeError(
                    "The gradient of the polynomial part must be a numpy array"
                )
            if not isinstance(int_grad2, np.ndarray):
                raise TypeError(
                    "The gradient of the polynomial part must be a numpy array"
                )
            self.int_grad1 += int_grad1
            self.int_grad2 += int_grad2

        # logarithmic part
        for k in range(self.mesh_cell.num_holes):
            xi = self.mesh_cell.components[k + 1].interior_point
            y_xi_1 = y1 - xi.x
            y_xi_2 = y2 - xi.y
            y_xi_norm_sq = y_xi_1**2 + y_xi_2**2
            self.int_vals += 0.5 * self.harm.log_coef[k] * np.log(y_xi_norm_sq)
            if compute_int_grad:
                self.int_grad1 += self.harm.log_coef[k] * y_xi_1 / y_xi_norm_sq
                self.int_grad2 += self.harm.log_coef[k] * y_xi_2 / y_xi_norm_sq

        # conjugable part
        psi = self.harm.psi.values
        psi_hat = self.harm.conj_trace.values

        # boundary points
        bdy_x1, bdy_x2 = self.mesh_cell.get_boundary_points()

        # shifted coordinates
        M = self.mesh_cell.num_pts
        xy1 = np.reshape(bdy_x1, (1, M)) - np.reshape(y1, (N, 1))
        xy2 = np.reshape(bdy_x2, (1, M)) - np.reshape(y2, (N, 1))
        xy_norm_sq = xy1**2 + xy2**2

        # components of unit tangent vector
        t1, t2 = self.mesh_cell.get_unit_tangent()

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
        dx_norm = self.mesh_cell.get_dx_norm()
        h = 2 * np.pi * self.mesh_cell.num_edges / self.mesh_cell.num_pts
        dx_norm *= h

        # interior values and gradient of conjugable part via Cauchy's
        # integral formula
        self.int_vals += np.sum(dx_norm * f, axis=1) * 0.5 / np.pi
        if compute_int_grad:
            self.int_grad1 += np.sum(dx_norm * g1, axis=1) * 0.5 / np.pi
            self.int_grad2 += np.sum(dx_norm * g2, axis=1) * 0.5 / np.pi
