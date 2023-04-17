import numpy as np
from ..mesh.cell import cell
from .locfun import locfun

def interior_values(v: locfun, K: cell):
	"""
	Returns (y1, y2, vals) where y1,y2 form a meshgrid covering the cell K
	and vals is an array of the same size of the interior values of v.
	At points that are not in the interior, vals is nan.
	"""

	y1, y2, is_inside = generate_interior_points(K)
	rows, cols = np.shape(y1)

	vals = np.zeros((rows, cols))
	grad1 = np.zeros((rows, cols))
	grad2 = np.zeros((rows, cols))

	# conjugable part
	psi = v.get_conjugable_part(K)
	psi_hat = v.get_harmonic_conjugate()
	x1, x2 = K.get_boundary_points()

	# polynomial gradient
	Px, Py = v.poly_part.grad()

	# compute interior values
	for i in range(rows):
		for j in range(cols):
			if is_inside[i, j]:

				# Cauchy's integral formula
				xy1 = x1 - y1[i, j]
				xy2 = x2 - y2[i, j]
				xy_norm_sq = xy1 * xy1 + xy2 * xy2
				eta = (xy1 * psi + xy2 * psi_hat) / xy_norm_sq
				eta_hat = (xy1 * psi_hat - xy2 * psi) / xy_norm_sq
				integrand = K.dot_with_tangent(eta_hat, eta)
				vals[i, j] = K.integrate_over_boundary(integrand) * 0.5 / np.pi

				# polynomial part
				vals[i, j] += v.poly_part.eval(y1[i, j], y2[i, j])

				# logarithmic part
				for k in range(K.num_holes):
					y_xi_norm_sq = (y1[i, j] - K.hole_int_pts[0, k]) ** 2 + \
						(y2[i, j] - K.hole_int_pts[1, k]) ** 2
					vals[i, j] += 0.5 * v.log_coef[k] * np.log(y_xi_norm_sq)

				# Cauchy's integral formula for gradient
				omega = (xy1 * eta + xy2 * eta_hat) / xy_norm_sq
				omega_hat = (xy1 * eta_hat - xy2 * eta) / xy_norm_sq
				integrand = K.dot_with_tangent(omega_hat, omega)
				grad1[i, j] = K.integrate_over_boundary(integrand) * 0.5 / np.pi
				integrand = K.dot_with_tangent(omega, -omega_hat)
				grad2[i, j] = K.integrate_over_boundary(integrand) * 0.5 / np.pi

				# gradient polynomial part
				grad1[i, j] += Px.eval(y1[i, j], y2[i, j])
				grad2[i, j] += Py.eval(y1[i, j], y2[i, j])

				# gradient logarithmic part
				for k in range(K.num_holes):
					y_xi_1 = y1[i, j] - K.hole_int_pts[0, k]
					y_xi_2 = y2[i, j] - K.hole_int_pts[1, k]
					y_xi_norm_sq = y_xi_1 ** 2 + y_xi_2 ** 2
					grad1[i, j] += v.log_coef[k] * y_xi_1 / y_xi_norm_sq
					grad2[i, j] += v.log_coef[k] * y_xi_2 / y_xi_norm_sq

			else:
				vals[i, j] = np.nan
				grad1[i, j] = np.nan
				grad2[i, j] = np.nan

	return y1, y2, vals, grad1, grad2

def generate_interior_points(K: cell, rows=101, cols=101, tol=0.02):
	"""
	Returns (x, y, is_inside) where x,y are a meshgrid covering the
	cell K, and is_inside is a boolean array that is True for
	interior points
	"""

	# find region of interest
	xmin, xmax, ymin, ymax = K._get_bounding_box()

	# set up grid
	x_coord = np.linspace(xmin, xmax, rows)
	y_coord = np.linspace(ymin, ymax, cols)
	x, y = np.meshgrid(x_coord, y_coord)

	# determine which points are inside K
	is_inside = K.is_in_interior_cell(x, y)

	# set minimum desired distance to the boundary
	TOL = tol * np.min([xmax - xmin, ymax - ymin])

	# ignore points too close to the boundary
	for i in range(rows):
		for j in range(cols):
			if is_inside[i, j]:
				d = K._get_distance_to_boundary(x[i, j], y[i, j])
				if d < TOL:
					is_inside[i, j] = False

	return x, y, is_inside