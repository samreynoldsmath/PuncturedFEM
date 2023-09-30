"""
planar_mesh.py
==============

Module containing the planar_mesh class, which handles planar mesh geometry and
topology.
"""

from .cell import cell
from .edge import edge


class planar_mesh:
    """
    Planar mesh geometry and topology. The edges of the mesh are permitted to be
    curved, and cells are permitted to be multiply connected.

    Edges meet at an interior angle strictly between 0 and 2 pi (no slits or
    cusps). Hanging nodes (i.e. an interior angle of pi) are permitted. Looped
    edges (edges with identical endpoints) are permitted under the same
    restrictions.

    The mesh geometry is determined by the parameterization of the edges. The
    mesh topology is automatically determined by observing that each edge is
    shared by the boundary of exactly two cells.

    Computations on the mesh dictate that the boundary of each cell have an
    oriented boundary (counterclockwise on the outer boundary, clockwise on the
    inner boundary). Rather than storing the parameterization of each edge
    twice (once for each cell), we record the orientation of each edge relative
    to the boundary of each cell.

    For instance, if the edge is oriented counterclockwise to a cell boundary,
    with that edge lying on the outer boundary of this cell, then the edge is
    said to have positive orientation with respect to that cell. Each edge
    object stores the cell index of the cell on either side of the edge, one
    positive and one negative. If the edge is on the boundary of the domain,
    the "missing" cell is assigned a negative cell index.

    Usage
    -----
    See examples/ex0-mesh-building.ipynb for an example of how to build a mesh.

    Attributes
    ----------
    num_edges : int
        Number of edges in the mesh.
    num_cells : int
        Number of cells in the mesh.
    num_verts : int
        Number of vertices in the mesh.
    edges : list[edge]
        List of edges in the mesh.
    vert_idx_list : list[int]
        List of vertex indices in the mesh, excluding vertices of loops.
    cell_idx_list : list[int]
        List of cell indices in the mesh.

    Notes
    -----
    - The vertex indices are taken to be nonnegative integers. These indices
      are not necessarily 0, 1, ...., num_verts - 1, but instead are determined
      by the indices assigned to the endpoints of the edges.
    - The cell indices are taken to be nonnegative integers, except for the case
      of edges on the boundary of the domain. In this case, the cell index is
      assigned to be negative. Similar to the vertex indices, the cell indices
      are not necessarily 0, 1, ...., num_cells - 1, but instead are determined
      by the indices assigned to the edges in the pos_cell_idx and neg_cell_idx
      attributes.
    """

    num_edges: int
    num_cells: int
    num_verts: int
    edges: list[edge]
    vert_idx_list: list[int]  # TODO: use set instead of list
    cell_idx_list: list[int]  # TODO: use set instead of list

    def __init__(self, edges: list[edge], verbose: bool = True) -> None:
        """
        Constructor for planar_mesh class.

        Parameters
        ----------
        edges : list[edge]
            List of edges in the mesh.
        verbose : bool, optional
            If True, print information about the mesh. Default is True.
        """
        if verbose:
            print("Building planar mesh...")
        self.edges = []
        self.num_edges = 0
        self.add_edges(edges)
        self.set_edge_ids()
        self.build_cell_idx_list()
        self.build_vert_idx_list()
        self.find_repeats()
        if verbose:
            print(self)

    def __str__(self) -> str:
        """
        String representation of planar_mesh object.
        """
        s = "planar_mesh:"
        s += f"\n\tnum_verts: {self.num_verts}"
        s += f"\n\tnum_edges: {self.num_edges}"
        s += f"\n\tnum_cells: {self.num_cells}"
        return s

    # EDGES ##################################################################
    def compute_num_edges(self) -> None:
        """
        Compute and store the number of edges in the mesh to self.num_edges.
        """
        self.num_edges = len(self.edges)

    def add_edge(self, e: edge) -> None:
        """Add an edge to the mesh."""
        if not isinstance(e, edge):
            raise TypeError("e must be an edge")
        if e in self.edges:
            return
        self.edges.append(e)
        self.num_edges += 1

    def add_edges(self, edges: list[edge]) -> None:
        """Add a list of edges to the mesh."""
        for e in edges:
            self.add_edge(e)

    def set_edge_ids(self) -> None:
        """Set the id of each edge in the mesh."""
        for k, e in enumerate(self.edges):
            e.set_idx(k)

    # VERTICES ###############################################################

    def compute_num_verts(self) -> None:
        """
        Compute and store the number of vertices in the mesh to self.num_verts.
        """
        self.num_verts = len(self.vert_idx_list)

    def build_vert_idx_list(self) -> None:
        """
        Build and store the list of vertex indices (excluding loops) in the
        mesh to self.vert_idx_list.
        """
        self.vert_idx_list = []
        for e in self.edges:
            if not e.is_loop:
                for vert_idx in [e.anchor.idx, e.endpnt.idx]:
                    if vert_idx >= 0 and vert_idx not in self.vert_idx_list:
                        self.vert_idx_list.append(vert_idx)
        self.vert_idx_list.sort()
        self.compute_num_verts()

    # CELLS ##################################################################

    def compute_num_cells(self) -> None:
        """
        Compute and store the number of cells in the mesh to self.num_cells.
        """
        self.num_cells = len(self.cell_idx_list)

    def build_cell_idx_list(self) -> None:
        """
        Build and store the list of cell indices in the mesh to
        self.cell_idx_list.
        """
        self.cell_idx_list = []
        for e in self.edges:
            for cell_idx in [e.pos_cell_idx, e.neg_cell_idx]:
                if cell_idx >= 0 and cell_idx not in self.cell_idx_list:
                    self.cell_idx_list.append(cell_idx)
        self.cell_idx_list.sort()
        self.compute_num_cells()

    def get_cell(self, cell_idx: int) -> cell:
        """
        Return the cell with index cell_idx.
        """
        if cell_idx not in self.cell_idx_list:
            raise Exception("cell_idx is not in cell_idx_list")
        edges = []
        for e in self.edges:
            if cell_idx in (e.pos_cell_idx, e.neg_cell_idx):
                edges.append(e)
        return cell(idx=cell_idx, edges=edges)

    def get_abs_cell_idx(self, cell_idx: int) -> int:
        """
        Return the absolute cell index of cell_idx.
        """
        return self.cell_idx_list.index(cell_idx)

    # BOUNDARY IDENTIFICATION ################################################

    def vert_is_on_boundary(self, vert_idx: int) -> bool:
        """
        Return True if the vertex with index vert_idx is on the boundary of the
        domain.
        """
        for e in self.edges:
            if vert_idx in (e.anchor.idx, e.endpnt.idx):
                if e.pos_cell_idx < 0 or e.neg_cell_idx < 0:
                    return True
        return False

    def edge_is_on_boundary(self, edge_idx: int) -> bool:
        """
        Return True if the edge with index edge_idx is on the boundary of the
        domain.
        """
        e = self.edges[edge_idx]
        return e.pos_cell_idx < 0 or e.neg_cell_idx < 0

    # REPEAT DETECTION #######################################################

    def find_repeats(self) -> None:
        """
        TODO: Find edges and cells that are repeated
        """
