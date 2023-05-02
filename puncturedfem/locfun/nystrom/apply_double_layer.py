def apply_T2(u, T2, T2_sum, closest_vert_idx):
	corner_values = u[closest_vert_idx]
	return 0.5 * (u - corner_values) + T2 @ u - corner_values * T2_sum