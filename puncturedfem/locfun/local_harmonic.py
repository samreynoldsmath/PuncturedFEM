from typing import Optional

import numpy as np

from . import antilap, fft_deriv
from .nystrom import NystromSolver
from .trace import DirichletTrace


class LocalHarmonic:
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
        self.set_trace(trace)
        self._compute_harmonic_conjugate(nyst)
        self._compute_harmonic_weighted_normal_derivative(nyst)
        if compute_biharmonic:
            self._compute_biharmonic(nyst)

    def set_trace(self, trace: DirichletTrace) -> None:
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
        harm_part_wnd = (
            fft_deriv.get_weighted_tangential_derivative_from_trace(
                nyst.K, self.conj_trace.values
            )
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
