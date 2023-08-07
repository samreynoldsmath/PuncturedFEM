from .poly import polynomial
from ...mesh.cell import cell

def integrate_poly_over_cell(p: polynomial, K: cell):
		""""
		Returns the value of
			\int_K (self) dx
		by reducing this volumetric integral to one on the boundary via
		the Divergence Theorem
		"""
		x1, x2 = K.get_boundary_points()
		xn = K.dot_with_normal(x1, x2)
		val = 0
		for m in p.monos:
			integrand = xn * m.eval(x1, x2) / (2 + m.alpha.order)
			val += K.integrate_over_boundary(integrand)
		return val