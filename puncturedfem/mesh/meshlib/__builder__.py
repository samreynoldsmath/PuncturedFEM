from ..planar_mesh import planar_mesh

def mesh_builder(get_verts: callable, get_edges:callable, verbose: bool=True,
	    **kwargs) -> planar_mesh:

	# define vertices
	verts = get_verts(**kwargs)

	# TODO: set vertex ids here?
	for k in range(len(verts)):
		verts[k].set_id(k)

	# define edges
	edges = get_edges(verts, **kwargs)

	# return planar mesh
	return planar_mesh(edges, verbose=verbose)