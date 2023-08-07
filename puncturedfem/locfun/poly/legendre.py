from numpy import sqrt
from .poly import polynomial

def legendre_polynomials(deg: int) -> list[polynomial]:
	"""
	Returns an array of polynomial objects
		[p_0, p_1, ..., p_deg]
	where p_j is the jth Legendre polynomial
	"""
	polys = []
	polys.append(polynomial([[1.0, 0, 0]]))
	polys.append(polynomial([[1.0, 1, 0]]))
	for n in range(2, deg + 1):
		pn = ((2 * n - 1) * polys[1] * polys[n - 1] \
			- (n - 1) * polys[n - 2]) / n
		polys.append(pn)
	return polys

def legendre_tensor_products(deg: int) -> list[polynomial]:
	p = legendre_polynomials(deg)
	q = []
	for pn in p:
		qn = swap_coordinates_of_poly_argument(pn)
		q.append(qn)
	pq = []
	for m in range(deg + 1):
		for n in range(deg + 1 - m):
			pq.append(p[m] * q[n])
	return pq

def integrated_legendre_polynomials(deg: int) -> list[polynomial]:
	polys = []
	p = legendre_polynomials(deg)
	polys.append(p[0])
	polys.append(p[1])
	for n in range(2, deg + 1):
		qn = (p[n] - p[n - 2]) / sqrt(4 * n - 2)
		polys.append(qn)
	return polys

def integrated_legendre_tensor_products(deg: int) -> list[polynomial]:
	p = integrated_legendre_polynomials(deg)
	q = []
	for pn in p:
		qn = swap_coordinates_of_poly_argument(pn)
		q.append(qn)
	pq = []
	for m in range(deg + 1):
		for n in range(deg + 1 - m):
			pq.append(p[m] * q[n])
	return pq

def swap_coordinates_of_poly_argument(p: polynomial):
	"""
	Returns the polynomial q(x,y) = p(y,x)
	"""
	q = polynomial()
	for mono in p.monos:
		coef = mono.coef
		alpha_0 = mono.alpha.x
		alpha_1 = mono.alpha.y
		q += polynomial([[coef, alpha_1, alpha_0]])
	return q