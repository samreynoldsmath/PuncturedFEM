"""
Elements of the local Poisson space V_p(K).

Classes
-------
LocalPoissonFunction
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ..solver.globkey import GlobalKey
from ..util.types import FloatLike
from .nystrom import NystromSolver
from .poly.poly import Polynomial
from .trace import DirichletTrace
from .local_harmonic import LocalHarmonic
from .local_polynomial import LocalPolynomial
from .inner_prod import h1_semi_inner_prod, l2_inner_prod
from ..mesh.cell import MeshCell


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
    harm : LocalHarmonic
        The harmonic part, phi.
    poly : LocalPolynomial
        The polynomial part, P.
    key : GlobalKey
        A unique tag that identifies the local function in the global space.
    """

    harm: LocalHarmonic
    poly: LocalPolynomial
    key: GlobalKey

    def __init__(
        self,
        nyst: NystromSolver,
        laplacian: Polynomial = Polynomial(),
        trace: Union[DirichletTrace, FloatLike] = 0,
        compute_for_l2: bool = True,
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
        P = laplacian.anti_laplacian()
        self.poly = LocalPolynomial(P, nyst.K)
        if isinstance(trace, (float, int, np.ndarray)):
            trace = DirichletTrace(edges=nyst.K.get_edges(), values=trace)
        harm_trace_vals = DirichletTrace(
            edges=nyst.K.get_edges(),
            values=trace.values - self.poly.trace.values,
        )
        self.harm = LocalHarmonic(harm_trace_vals, nyst, compute_for_l2)

    def get_h1_semi_inner_prod(
        self,
        other: Union[LocalPoissonFunction, LocalHarmonic, LocalPolynomial],
        K: MeshCell,
    ) -> float:
        """
        Return the H^1 semi-inner product int_K grad(self) * grad(other) dx.

        Parameters
        ----------
        other : LocalPoissonFunction, LocalHarmonic, or LocalPolynomial
            Another local function on the same mesh cell.
        K : MeshCell
            The mesh cell on which the functions are defined.
        """
        if isinstance(other, LocalPoissonFunction):
            val = h1_semi_inner_prod(self.harm, other.harm, K)
            val += h1_semi_inner_prod(self.poly, other.poly, K)
            val += h1_semi_inner_prod(self.harm, other.poly, K)
            val += h1_semi_inner_prod(self.poly, other.harm, K)
            return val
        if isinstance(other, (LocalHarmonic, LocalPolynomial)):
            val = h1_semi_inner_prod(self.harm, other, K)
            val += h1_semi_inner_prod(self.poly, other, K)
            return val

    def get_l2_inner_prod(
        self,
        other: Union[LocalPoissonFunction, LocalHarmonic, LocalPolynomial],
        K: MeshCell,
    ) -> float:
        """
        Return the L^2 inner product int_K self * other dx.

        Parameters
        ----------
        other : LocalPoissonFunction, LocalHarmonic, or LocalPolynomial
            Another local function on the same mesh cell.
        K : MeshCell
            The mesh cell on which the functions are defined.
        """
        if isinstance(other, LocalPoissonFunction):
            val = l2_inner_prod(self.harm, other.harm, K)
            val += l2_inner_prod(self.harm, other.poly, K)
            val += l2_inner_prod(self.poly, other.harm, K)
            val += l2_inner_prod(self.poly, other.poly, K)
            return val
        if isinstance(other, (LocalHarmonic, LocalPolynomial)):
            val = l2_inner_prod(self.harm, other, K)
            val += l2_inner_prod(self.poly, other, K)
            return val

    def _set_key(self, key: Optional[GlobalKey]) -> None:
        if key is None:
            return
        if not isinstance(key, GlobalKey):
            raise TypeError("key must be a GlobalKey")
        self.key = key
