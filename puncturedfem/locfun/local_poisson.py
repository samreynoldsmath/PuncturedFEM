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
    key : GlobalKey
        A unique tag that identifies the local function in the global space.
    """

    harm: LocalHarmonic
    poly: LocalPolynomial
    mesh_cell: MeshCell
    key: GlobalKey

    def __init__(
        self,
        nyst: NystromSolver,
        laplacian: Polynomial = Polynomial(),
        trace: Union[DirichletTrace, FloatLike] = 0,
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
        key : Optional[GlobalKey], optional
            A unique tag that identifies the local function in the global space.
        """
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
