"""
Inner products, including H^1 semi-inner products and L^2 inner products.

Functions
---------
h1_semi_inner_prod(v, w, K)
    Return the H^1 semi-inner product int_K grad(v) * grad(w) dx.
l2_inner_prod(v, w, K)
    Return the L^2 inner product int_K (v) * (w) dx.
"""

from typing import Union

from ..mesh.cell import MeshCell
from .local_harmonic import LocalHarmonic
from .local_polynomial import LocalPolynomial
from .poly.integrate_poly import integrate_poly_over_mesh_cell

LocalGeneric = Union[LocalHarmonic, LocalPolynomial]


def h1_semi_inner_prod(v: LocalGeneric, w: LocalGeneric, K: MeshCell) -> float:
    """
    Return the H^1 semi-inner product int_K grad(v) * grad(w) dx.

    Parameters
    ----------
    v : Union[LocalHarmonic, LocalPolynomial]
        A LocalHarmonic or LocalPolynomial.
    w : Union[LocalHarmonic, LocalPolynomial]
        Another LocalHarmonic or LocalPolynomial.
    K : MeshCell
        The mesh cell over which to integrate.

    Returns
    -------
    float
        The H^1 semi-inner product int_K grad(v) * grad(w) dx.
    """
    if not isinstance(v, (LocalHarmonic, LocalPolynomial)):
        raise TypeError("v must be a LocalHarmonic or LocalPolynomial")
    if not isinstance(w, (LocalHarmonic, LocalPolynomial)):
        raise TypeError("w must be a LocalHarmonic or LocalPolynomial")
    if not isinstance(K, MeshCell):
        raise TypeError("K must be a MeshCell")
    if isinstance(v, LocalHarmonic):
        if isinstance(w, LocalHarmonic):
            return _h1_semi_inner_prod_harmonic_harmonic(v, w, K)
        return _h1_semi_inner_prod_harmonic_polynomial(v, w, K)
    if isinstance(w, LocalHarmonic):
        return _h1_semi_inner_prod_harmonic_polynomial(w, v, K)
    return _h1_semi_inner_prod_polynomial_polynomial(v, w, K)


def l2_inner_prod(v: LocalGeneric, w: LocalGeneric, K: MeshCell) -> float:
    """
    Return the L^2 inner product int_K (v) * (w) dx.

    Parameters
    ----------
    v : Union[LocalHarmonic, LocalPolynomial]
        A LocalHarmonic or LocalPolynomial.
    w : Union[LocalHarmonic, LocalPolynomial]
        Another LocalHarmonic or LocalPolynomial.
    K : MeshCell
        The mesh cell over which to integrate.

    Returns
    -------
    float
        The L^2 inner product int_K (v) * (w) dx.
    """
    if not isinstance(v, (LocalHarmonic, LocalPolynomial)):
        raise TypeError("v must be a LocalHarmonic or LocalPolynomial")
    if not isinstance(w, (LocalHarmonic, LocalPolynomial)):
        raise TypeError("w must be a LocalHarmonic or LocalPolynomial")
    if not isinstance(K, MeshCell):
        raise TypeError("K must be a MeshCell")
    if isinstance(v, LocalHarmonic):
        if isinstance(w, LocalHarmonic):
            return _l2_inner_prod_harmonic_harmonic(v, w, K)
        return _l2_inner_prod_harmonic_polynomial(v, w, K)
    if isinstance(w, LocalHarmonic):
        return _l2_inner_prod_harmonic_polynomial(w, v, K)
    return _l2_inner_prod_polynomial_polynomial(v, w, K)


def _h1_semi_inner_prod_harmonic_harmonic(
    phi: LocalHarmonic, psi: LocalHarmonic, K: MeshCell
) -> float:
    return K.integrate_over_boundary_preweighted(
        phi.trace.w_norm_deriv * psi.trace.values
    )


def _h1_semi_inner_prod_harmonic_polynomial(
    phi: LocalHarmonic, poly: LocalPolynomial, K: MeshCell
) -> float:
    return K.integrate_over_boundary_preweighted(
        phi.trace.w_norm_deriv * poly.trace.values
    )


def _h1_semi_inner_prod_polynomial_polynomial(
    poly1: LocalPolynomial, poly2: LocalPolynomial, K: MeshCell
) -> float:
    return integrate_poly_over_mesh_cell(
        poly1.grad1 * poly2.grad1 + poly1.grad2 * poly2.grad2, K
    )


def _l2_inner_prod_harmonic_harmonic(
    phi: LocalHarmonic, psi: LocalHarmonic, K: MeshCell
) -> float:
    if phi.biharmonic_trace is None or psi.biharmonic_trace is None:
        raise ValueError("Both harmonics must have biharmonic traces")
    val = K.integrate_over_boundary_preweighted(
        phi.biharmonic_trace.w_norm_deriv * psi.trace.values
    )
    val -= K.integrate_over_boundary_preweighted(
        phi.biharmonic_trace.values * psi.trace.w_norm_deriv
    )
    return val


def _l2_inner_prod_harmonic_polynomial(
    phi: LocalHarmonic, poly: LocalPolynomial, K: MeshCell
) -> float:
    if poly.antilap_trace.w_norm_deriv is None:
        raise ValueError("Polynomial must have weighted normal derivative")
    val = K.integrate_over_boundary_preweighted(
        phi.trace.values * poly.antilap_trace.w_norm_deriv
    )
    val -= K.integrate_over_boundary_preweighted(
        phi.trace.w_norm_deriv * poly.antilap_trace.values
    )
    return val


def _l2_inner_prod_polynomial_polynomial(
    poly1: LocalPolynomial, poly2: LocalPolynomial, K: MeshCell
) -> float:
    PQ = poly1.exact_form * poly2.exact_form
    return integrate_poly_over_mesh_cell(PQ, K)
