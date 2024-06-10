"""
Harmonic function on a mesh cell.

Classes
-------
LocalHarmonic
    Harmonic function on a mesh cell.
"""

from __future__ import annotations
from typing import Optional, Union

import numpy as np

from .antilap import get_anti_laplacian_harmonic
from .fft_deriv import get_weighted_tangential_derivative_from_trace
from .nystrom import NystromSolver
from .trace import DirichletTrace


class LocalHarmonic:
    """
    Harmonic function on a mesh cell.

    The harmonic function is represented by its Dirichlet trace. A harmonic
    function has a harmonic conjugate on simply connected mesh cells. On
    multiply connected mesh cells, the harmonic conjugate does not generally
    exist, but the harmonic function can be decomposed into a "conjugable part"
    and a sum of logarithmic terms.

    In order to compute the L^2 inner product of two harmonic functions, we
    compute a biharmonic function whose Laplacian is taken to be the given
    harmonic function. The biharmonic function is represented by its Dirichlet
    trace.

    The weighted normal derivative of the harmonic function is also computed,
    and stored in self.trace.w_norm_deriv, and likewise for the biharmonic
    function.

    Attributes
    ----------
    trace : DirichletTrace
        Dirichlet trace of the local harmonic function.
    psi : DirichletTrace
        Dirichlet trace of the conjugable part of the local harmonic function.
    conj_trace : DirichletTrace
        Dirichlet trace of the harmonic conjugate of the local harmonic
        function.
    log_coef : list[float]
        Coefficients of the logarithmic terms in the harmonic conjugate.
    biharmonic_trace : Optional[DirichletTrace]
        Dirichlet trace of the biharmonic part of the local harmonic function.
    """

    trace: DirichletTrace
    psi: DirichletTrace
    conj_trace: DirichletTrace
    log_coef: list[float]
    biharmonic_trace: Optional[DirichletTrace]

    def __init__(
        self, trace: DirichletTrace, nyst: Optional[NystromSolver]
    ) -> None:
        """
        Initialize the local harmonic function.

        Parameters
        ----------
        trace : DirichletTrace
            Dirichlet trace of the local harmonic function.
        nyst : Optional[NystromSolver]
            Nystrom solver for the mesh cell.
        """
        self.set_trace(trace)
        if nyst is None:
            return
        self.trace.set_weighted_tangential_derivative(
            get_weighted_tangential_derivative_from_trace(nyst.K, trace.values)
        )
        self._compute_harmonic_conjugate(nyst)
        self._compute_conjugable_part(nyst)
        self._compute_harmonic_weighted_normal_derivative(nyst)
        self._compute_biharmonic(nyst)

    def __add__(self, other: LocalHarmonic) -> LocalHarmonic:
        """
        Add two local harmonic functions.

        Parameters
        ----------
        other : LocalHarmonic
            The other local harmonic function.

        Returns
        -------
        LocalHarmonic
            The sum of the two local harmonic functions.
        """
        new = LocalHarmonic(trace=self.trace + other.trace, nyst=None)
        new.psi = self.psi + other.psi
        new.conj_trace = self.conj_trace + other.conj_trace
        new.log_coef = [a + b for a, b in zip(self.log_coef, other.log_coef)]
        if self.biharmonic_trace is not None and other.biharmonic_trace is not None:
            new.biharmonic_trace = self.biharmonic_trace + other.biharmonic_trace
        return new

    def __mul__(self, other: Union[int, float]) -> LocalHarmonic:
        """
        Multiply the local harmonic function by a scalar.

        Parameters
        ----------
        other : Union[int, float]
            The scalar.

        Returns
        -------
        LocalHarmonic
            The product of the local harmonic function and the scalar.
        """
        new = LocalHarmonic(trace=self.trace * other, nyst=None)
        new.psi = self.psi * other
        new.conj_trace = self.conj_trace * other
        new.log_coef = [a * other for a in self.log_coef]
        if self.biharmonic_trace is not None:
            new.biharmonic_trace = self.biharmonic_trace * other
        return new

    def __rmul__(self, other: Union[int, float]) -> LocalHarmonic:
        """
        Multiply the local harmonic function by a scalar.

        Parameters
        ----------
        other : Union[int, float]
            The scalar.

        Returns
        -------
        LocalHarmonic
            The product of the local harmonic function and the scalar.
        """
        return self.__mul__(other)

    def __sub__(self, other: LocalHarmonic) -> LocalHarmonic:
        """
        Subtract two local harmonic functions.

        Parameters
        ----------
        other : LocalHarmonic
            The other local harmonic function.

        Returns
        -------
        LocalHarmonic
            The difference of the two local harmonic functions.
        """
        return self + other * -1

    def __truediv__(self, other: Union[int, float]) -> LocalHarmonic:
        """
        Divide the local harmonic function by a scalar.

        Parameters
        ----------
        other : Union[int, float]
            The scalar.

        Returns
        -------
        LocalHarmonic
            The division of the local harmonic function by the scalar.
        """
        if not isinstance(other, (int, float)):
            raise TypeError("The divisor must be a scalar.")
        if other == 0:
            raise ValueError("Division by zero.")
        new = LocalHarmonic(trace=self.trace / other, nyst=None)
        new.psi = self.psi / other
        new.conj_trace = self.conj_trace / other
        new.log_coef = [a / other for a in self.log_coef]
        if self.biharmonic_trace is not None:
            new.biharmonic_trace = self.biharmonic_trace / other
        return new

    def set_trace(self, trace: DirichletTrace) -> None:
        """
        Set the Dirichlet trace of the local harmonic function.
        """
        if not isinstance(trace, DirichletTrace):
            raise TypeError("trace must be a DirichletTrace")
        self.trace = trace

    def _compute_harmonic_conjugate(self, nyst: NystromSolver) -> None:
        harm_conj_trace_values, log_coef = nyst.get_harmonic_conjugate(
            phi=self.trace.values
        )
        self.conj_trace = DirichletTrace(
            edges=nyst.K.get_edges(), values=harm_conj_trace_values
        )
        self.log_coef = list(log_coef)

    def _compute_harmonic_weighted_normal_derivative(
        self, nyst: NystromSolver
    ) -> None:
        harm_part_wnd = get_weighted_tangential_derivative_from_trace(
            nyst.K, self.conj_trace.values
        )
        for j in range(nyst.K.num_holes):
            harm_part_wnd += self.log_coef[j] * nyst.lam_trace[j].w_norm_deriv
        self.trace.set_weighted_normal_derivative(harm_part_wnd)

    def _compute_conjugable_part(self, nyst: NystromSolver) -> None:
        lam = np.zeros((nyst.K.num_pts,))
        for j in range(nyst.K.num_holes):
            lam += self.log_coef[j] * nyst.lam_trace[j].values
        self.psi = DirichletTrace(
            edges=nyst.K.get_edges(), values=self.trace.values - lam
        )

    def _compute_biharmonic(self, nyst: NystromSolver) -> None:
        psi = self.psi.values
        psi_hat = self.conj_trace.values
        (
            big_phi,
            big_phi_wnd,
        ) = get_anti_laplacian_harmonic(
            nyst, psi, psi_hat, np.array(self.log_coef)
        )
        self.biharmonic_trace = DirichletTrace(
            edges=nyst.K.get_edges(), values=big_phi
        )
        self.biharmonic_trace.set_weighted_normal_derivative(big_phi_wnd)
