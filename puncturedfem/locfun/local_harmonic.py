"""
Harmonic function on a mesh cell.

Classes
-------
LocalHarmonic
    Harmonic function on a mesh cell.
"""

from typing import Optional

import numpy as np

from . import antilap, fft_deriv
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
    conj_trace : DirichletTrace
        Dirichlet trace of the harmonic conjugate of the local harmonic
        function.
    log_coef : list[float]
        Coefficients of the logarithmic terms in the harmonic conjugate.
    biharmonic_trace : Optional[DirichletTrace]
        Dirichlet trace of the biharmonic part of the local harmonic function.
    """

    trace: DirichletTrace
    conj_trace: DirichletTrace
    log_coef: list[float]
    biharmonic_trace: Optional[DirichletTrace]

    def __init__(
        self,
        trace: DirichletTrace,
        nyst: NystromSolver,
        compute_biharmonic: bool = True,
    ) -> None:
        """
        Initialize the local harmonic function.

        Parameters
        ----------
        trace : DirichletTrace
            Dirichlet trace of the local harmonic function.
        nyst : NystromSolver
            Nystrom solver object for solving integral equations.
        compute_biharmonic : bool, optional
            Whether to compute the biharmonic part of the local harmonic
            function, by default True.
        """
        self.set_trace(trace)
        self._compute_harmonic_conjugate(nyst)
        self._compute_harmonic_weighted_normal_derivative(nyst)
        if compute_biharmonic:
            self._compute_biharmonic(nyst)

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
        harm_part_wnd = fft_deriv.get_weighted_tangential_derivative_from_trace(
            nyst.K, self.conj_trace.values
        )
        for j in range(nyst.K.num_holes):
            harm_part_wnd += self.log_coef[j] * nyst.lam_trace[j].w_norm_deriv
        self.trace.set_weighted_normal_derivative(harm_part_wnd)

    def _get_conjugable_part(self, nyst: NystromSolver) -> np.ndarray:
        lam = np.zeros((nyst.K.num_pts,))
        for j in range(nyst.K.num_holes):
            lam += self.log_coef[j] * nyst.lam_trace[j].values
        return self.trace.values - lam

    def _compute_biharmonic(self, nyst: NystromSolver) -> None:
        psi = self._get_conjugable_part(nyst)
        psi_hat = self.conj_trace.values
        (
            big_phi,
            big_phi_wnd,
        ) = antilap.get_anti_laplacian_harmonic(
            nyst, psi, psi_hat, np.array(self.log_coef)
        )
        self.biharmonic_trace = DirichletTrace(
            edges=nyst.K.get_edges(), values=big_phi
        )
        self.biharmonic_trace.set_weighted_normal_derivative(big_phi_wnd)
