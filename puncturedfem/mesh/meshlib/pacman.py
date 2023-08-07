from numpy import sqrt
from ..vert import vert
from ..edge import edge
from ..planar_mesh import planar_mesh
from .__builder__ import mesh_builder

ROOT3OVER2 = sqrt(3) / 2
PACMAN_SCALE = 0.4
PACMAN_XSHIFT = 0.5
PACMAN_YSHIFT = 0.5
GHOST_SCALE = 0.6
GHOST_X_SHIFT = 3.5
GHOST_Y_SHIFT = 0.5

def pacman(verbose: bool=True) -> planar_mesh:
	return mesh_builder(get_verts, get_edges, verbose=verbose)

### VERTICES ###################################################################

def get_verts() -> list[vert]:

	verts: list[vert] = []

	# rectangle corners
	verts.append(vert(x=0.0, y=0.0)) # 0
	verts.append(vert(x=1.0, y=0.0)) # 1
	verts.append(vert(x=3.0, y=0.0)) # 2
	verts.append(vert(x=4.0, y=0.0)) # 3
	verts.append(vert(x=4.0, y=1.0)) # 4
	verts.append(vert(x=3.0, y=1.0)) # 5
	verts.append(vert(x=1.0, y=1.0)) # 6
	verts.append(vert(x=0.0, y=1.0)) # 7

	# "Pac-Man"
	verts.append(vert(x=0.5, y=0.5)) # 8
	verts.append(vert(
		x=PACMAN_XSHIFT + PACMAN_SCALE * ROOT3OVER2,
		y=PACMAN_YSHIFT + PACMAN_SCALE * 0.5)) # 9
	verts.append(vert(
		x=PACMAN_XSHIFT + PACMAN_SCALE * ROOT3OVER2,
		y=PACMAN_YSHIFT - PACMAN_SCALE * 0.5)) # 10
	verts.append(vert(
		x=PACMAN_XSHIFT + PACMAN_SCALE * -0.1,
		y=PACMAN_YSHIFT + PACMAN_SCALE * 0.5)) # 11

	# central "dots"
	verts.append(vert(x=1.5, y=0.5)) # 12
	verts.append(vert(x=2.0, y=0.5)) # 13
	verts.append(vert(x=2.5, y=0.5)) # 14

	# "ghost"
	verts.append(vert(
		x=GHOST_X_SHIFT + GHOST_SCALE * (-0.5),
		y=GHOST_Y_SHIFT + GHOST_SCALE * (-0.6))) # 15
	verts.append(vert(
		x=GHOST_X_SHIFT + GHOST_SCALE * (0.5),
		y=GHOST_Y_SHIFT + GHOST_SCALE * (-0.6))) # 16
	verts.append(vert(
		x=GHOST_X_SHIFT + GHOST_SCALE * (0.5),
		y=GHOST_Y_SHIFT + GHOST_SCALE * (0.2))) # 17
	verts.append(vert(
		x=GHOST_X_SHIFT + GHOST_SCALE * (-0.5),
		y=GHOST_Y_SHIFT + GHOST_SCALE * (0.2))) # 18
	verts.append(vert(
		x=GHOST_X_SHIFT + GHOST_SCALE * (-0.25),
		y=GHOST_Y_SHIFT + GHOST_SCALE * (0.1))) # 19
	verts.append(vert(
		x=GHOST_X_SHIFT + GHOST_SCALE * (0.25),
		y=GHOST_Y_SHIFT + GHOST_SCALE * (0.1))) # 20

	return verts

### EDGES ######################################################################

def get_edges(verts: list[vert]) -> list[edge]:

	# define edges
	edges = []

	# rectangles
	edges.append(edge(verts[0], verts[1], pos_cell_idx=0))
	edges.append(edge(verts[1], verts[2], pos_cell_idx=3))
	edges.append(edge(verts[2], verts[3], pos_cell_idx=7))
	edges.append(edge(verts[3], verts[4], pos_cell_idx=7))
	edges.append(edge(verts[4], verts[5], pos_cell_idx=7))
	edges.append(edge(verts[5], verts[6], pos_cell_idx=3))
	edges.append(edge(verts[6], verts[7], pos_cell_idx=0))
	edges.append(edge(verts[7], verts[0], pos_cell_idx=0))
	edges.append(edge(verts[1], verts[6], pos_cell_idx=0, neg_cell_idx=3))
	edges.append(edge(verts[2], verts[5], pos_cell_idx=3, neg_cell_idx=7))

	# pacman
	edges.append(edge(verts[8], verts[9], pos_cell_idx=1, neg_cell_idx=0))
	edges.append(edge(verts[9], verts[10], pos_cell_idx=1, neg_cell_idx=0,
		curve_type='circular_arc_deg',
		theta0=300))
	edges.append(edge(verts[10], verts[8], pos_cell_idx=1, neg_cell_idx=0))
	edges.append(edge(verts[11], verts[11], pos_cell_idx=2, neg_cell_idx=1,
		curve_type='circle',
		radius=0.25 * PACMAN_SCALE))

	# dots
	edges.append(edge(verts[12], verts[12], pos_cell_idx=4, neg_cell_idx=3,
		curve_type='circle',
		radius=0.1))
	edges.append(edge(verts[13], verts[13], pos_cell_idx=5, neg_cell_idx=3,
		curve_type='circle',
		radius=0.1))
	edges.append(edge(verts[14], verts[14], pos_cell_idx=6, neg_cell_idx=3,
		curve_type='circle',
		radius=0.1))

	# ghost
	edges.append(edge(verts[15], verts[16], pos_cell_idx=8, neg_cell_idx=7,
		curve_type='sine_wave',
		amp=0.1,
		freq=6))
	edges.append(edge(verts[16], verts[17], pos_cell_idx=8, neg_cell_idx=7))
	edges.append(edge(verts[17], verts[18], pos_cell_idx=8, neg_cell_idx=7,
		curve_type='circular_arc_deg',
		theta0=180))
	edges.append(edge(verts[18], verts[15], pos_cell_idx=8, neg_cell_idx=7))
	edges.append(edge(verts[19], verts[19], pos_cell_idx=9, neg_cell_idx=8,
		curve_type='ellipse',
		a=0.15 * GHOST_SCALE,
		b=0.2 * GHOST_SCALE))
	edges.append(edge(verts[20], verts[20], pos_cell_idx=10, neg_cell_idx=8,
		curve_type='ellipse',
		a=0.15 * GHOST_SCALE,
		b=0.2 * GHOST_SCALE))

	return edges