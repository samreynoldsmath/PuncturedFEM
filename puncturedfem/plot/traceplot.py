import numpy as np
import matplotlib.pyplot as plt

# from .. import quad
from ..mesh.cell import cell

def plot_trace(f_trace_list, fmt, legend, title, K: cell, quad_list):

	t = _get_trace_param_cell_boundary(K, quad_list)

	plt.figure()
	for k in range(len(f_trace_list)):
		plt.plot(t, f_trace_list[k], fmt[k])
	plt.legend(legend)
	plt.grid('minor')
	plt.title(title)

	x_ticks, x_labels = _get_ticks(K)
	plt.xticks(ticks=x_ticks, labels=x_labels)

	plt.show()

	return None

def plot_trace_log(f_trace_list, fmt, legend, title, K: cell, quad_list):

	t = _get_trace_param_cell_boundary(K, quad_list)

	plt.figure()
	for k in range(len(f_trace_list)):
		plt.semilogy(t, f_trace_list[k], fmt[k])
	plt.legend(legend)
	plt.grid('minor')
	plt.title(title)

	x_ticks, x_labels = _get_ticks(K)
	plt.xticks(ticks=x_ticks, labels=x_labels)

	plt.show()

	return None

def _make_quad_dict(quad_list):
	"""
	Organize a list of distinct quad objects into a convenient dictionary
	"""
	quad_dict = dict()
	for q in quad_list:
		quad_dict[q.type] = q
	return quad_dict

def _get_trace_param_cell_boundary(K: cell, quad_list):

	quad_dict = _make_quad_dict(quad_list)

	t = np.zeros((K.num_pts,))
	t0 = 0
	idx_start = 0
	for e in K.edge_list:
		t[idx_start:(idx_start + e.num_pts - 1)] = \
			t0 + quad_dict[e.qtype].t[:-1]
		idx_start += e.num_pts - 1
		t0 += 2 * np.pi

	return t

def _get_ticks(K):
	x_ticks = np.linspace(0, 2 * np.pi * K.num_edges, K.num_edges + 1)
	x_labels = ['0',]
	for k in range(1, K.num_edges+1):
		x_labels.append(f'{2 * k}$\pi$')
	return x_ticks, x_labels

