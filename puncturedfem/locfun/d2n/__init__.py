"""
d2n
===

Tools for computing the Dirichlet-to-Neumann map of a harmonic function on a
simply or multiply connected domain.

Modules
-------
fft_deriv
    Fourier differentiation.
log_terms
    Traces and derivatives of logarithmic terms.
trace2tangential
    Obtain a tangential derivative from a trace.
"""

from . import fft_deriv, log_terms, trace2tangential

__all__ = [
    "fft_deriv",
    "log_terms",
    "trace2tangential",
]
