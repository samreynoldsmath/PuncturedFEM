"""
Planar mesh geometry and topology.

Classes
-------
PlanarMesh
    Planar mesh geometry and topology.
"""

from .cell import MeshCell
from .edge import Edge


class PlanarMesh:
    """
    Planar mesh geometry and topology.

    The edges of the mesh are permitted to be curved, and mesh cells are
    permitted to be multiply connected.

    edges meet at an interior angle strictly between 0 and 2 pi (no slits or
    cusps). Hanging nodes (i.e. an interior angle of pi) are permitted. Looped
    edges (edges with identical endpoints) are permitted under the same
    restrictions.

    The mesh geometry is determined by the parameterization of the edges. The
    mesh topology is automatically determined by observing that each Edge is
    shared by the boundary of exactly two MeshCells.

    Computations on the mesh dictate that the boundary of each MeshCell have an
    oriented boundary (counterclockwise on the outer boundary, clockwise on the
    inner boundary). Rather than storing the parameterization of each Edge
    twice (once for each MeshCell), we record the orientation of each Edge
    relative to the boundary of each MeshCell.

    For instance, if the Edge is oriented counterclockwise to a MeshCell
    boundary, with that Edge lying on the outer boundary of this MeshCell, then
    the Edge is said to have positive orientation with respect to that MeshCell.
    Each Edge object stores the MeshCell index of the MeshCell on either side of
    the Edge, one positive and one negative. If the Edge is on the boundary of
    the domain, the "missing" MeshCell is assigned a negative MeshCell index.

    Usage
    -----
    See examples/ex0-mesh-building.ipynb for an example of how to build a mesh.

    Attributes
    ----------
    num_edges : int
        Number of edges in the mesh.
    num_cells : int
        Number of MeshCells in the mesh.
    num_verts : int
        Number of Vertices in the mesh.
    edges : list[Edge]
        List of edges in the mesh.
    vert_idx_list : list[int]
        List of vertex indices in the mesh, excluding Vertices of loops.
    cell_idx_list : list[int]
        List of MeshCell indices in the mesh.

    Notes
    -----
    - The vertex indices are taken to be nonnegative integers. These indices
      are not necessarily 0, 1, ...., num_verts - 1, but instead are determined
      by the indices assigned to the endpoints of the edges.
    - The MeshCell indices are taken to be nonnegative integers, except for the
      case of edges on the boundary of the domain. In this case, the MeshCell
      index is assigned to be negative. Similar to the vertex indices, the
      MeshCell indices are not necessarily 0, 1, ...., num_cells - 1, but
      instead are determined by the indices assigned to the edges in the
      pos_cell_idx and neg_cell_idx attributes.
    """

    num_edges: int
    num_cells: int
    num_verts: int
    edges: list[Edge]
    vert_idx_list: list[int]
    cell_idx_list: list[int]

    def __init__(self, edges: list[Edge], verbose: bool = True) -> None:
        """
        Initialize a PlanarMesh object.

        Parameters
        ----------
        edges : list[Edge]
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
        # self.find_repeats()
        if verbose:
            print(self)

    def __str__(self) -> str:
        """Return a string representation of the PlanarMesh object."""
        s = "PlanarMesh:"
        s += f"\n\tnum_verts: {self.num_verts}"
        s += f"\n\tnum_edges: {self.num_edges}"
        s += f"\n\tnum_cells: {self.num_cells}"
        return s

    # EDGES ##################################################################
    def compute_num_edges(self) -> None:
        """
        Compute and store the number of edges in the mesh to self.num_edges.

        The following attributes are set:
        - num_edges : int
            Number of edges in the mesh.
        """
        self.num_edges = len(self.edges)

    def add_edge(self, e: Edge) -> None:
        """
        Add an Edge to the mesh.

        Parameters
        ----------
        e : Edge
            Edge to add to the mesh.
        """
        if not isinstance(e, Edge):
            raise TypeError("e must be an Edge")
        if e.neg_cell_idx < 0 and e.pos_cell_idx < 0:
            raise ValueError(
                "Edge must be part of the boundary of "
                + "at least one mesh cell"
            )
        if e in self.edges:
            return
        self.edges.append(e)
        self.num_edges += 1

    def add_edges(self, edges: list[Edge]) -> None:
        """
        Add a list of edges to the mesh.

        Parameters
        ----------
        edges : list[Edge]
            List of edges to add to the mesh.
        """
        for e in edges:
            self.add_edge(e)

    def set_edge_ids(self) -> None:
        """
        Set the id of each Edge in the mesh.

        The following attributes are set:
        - Edge.idx : int
            Index of the Edge in the mesh.
        """
        for k, e in enumerate(self.edges):
            e.set_idx(k)

    # VERTICES ###############################################################

    def compute_num_verts(self) -> None:
        """
        Compute and store the number of vertices in the mesh.

        The following attributes are set:
        - num_verts : int
            Number of vertices in the mesh.
        """
        self.num_verts = len(self.vert_idx_list)

    def build_vert_idx_list(self) -> None:
        """
        Build and store the list of vertex indices (excluding loops).

        The following attributes are set:
        - vert_idx_list : list[int]
            List of vertex indices in the mesh, excluding vertices of loops.
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
        Compute and store the number of MeshCells in the mesh.

        The following attributes are set:
        - num_cells : int
            Number of MeshCells in the mesh.
        """
        self.num_cells = len(self.cell_idx_list)

    def build_cell_idx_list(self) -> None:
        """
        Build and store the list of MeshCell indices in the mesh.

        The following attributes are set:
        - cell_idx_list : list[int]
            List of MeshCell indices in the mesh.
        """
        self.cell_idx_list = []
        for e in self.edges:
            for cell_idx in [e.pos_cell_idx, e.neg_cell_idx]:
                if cell_idx >= 0 and cell_idx not in self.cell_idx_list:
                    self.cell_idx_list.append(cell_idx)
        self.cell_idx_list.sort()
        self.compute_num_cells()

    def get_cells(self, cell_idx: int) -> MeshCell:
        """
        Return the MeshCell with index cell_idx.

        Parameters
        ----------
        cell_idx : int
            Index of the MeshCell to return.

        Returns
        -------
        MeshCell
            MeshCell with index cell_idx.
        """
        if cell_idx not in self.cell_idx_list:
            raise IndexError("cell_idx is not in cell_idx_list")
        edges = []
        for e in self.edges:
            if cell_idx in (e.pos_cell_idx, e.neg_cell_idx):
                edges.append(e)
        return MeshCell(idx=cell_idx, edges=edges)

    def get_abs_cell_idx(self, cell_idx: int) -> int:
        """
        Return the absolute MeshCell index of cell_idx.

        Parameters
        ----------
        cell_idx : int
            Index of the MeshCell to return.

        Returns
        -------
        int
            Absolute index of the MeshCell.
        """
        return self.cell_idx_list.index(cell_idx)

    # BOUNDARY IDENTIFICATION ################################################

    def vert_is_on_boundary(self, vert_idx: int) -> bool:
        """
        Return True if the vertex with index vert_idx is on the boundary.

        Parameters
        ----------
        vert_idx : int
            Index of the vertex.

        Returns
        -------
        bool
            True if the vertex is on the boundary, False otherwise.
        """
        for e in self.edges:
            if vert_idx in (e.anchor.idx, e.endpnt.idx):
                if e.pos_cell_idx < 0 or e.neg_cell_idx < 0:
                    return True
        return False

    def edge_is_on_boundary(self, edge_idx: int) -> bool:
        """
        Return True if the Edge with index edge_idx is on the boundary.

        Parameters
        ----------
        edge_idx : int
            Index of the Edge.

        Returns
        -------
        bool
            True if the Edge is on the boundary, False otherwise.
        """
        e = self.edges[edge_idx]
        return e.pos_cell_idx < 0 or e.neg_cell_idx < 0

    # REPEAT DETECTION #######################################################

    def find_repeats(self) -> None:
        """
        Find edges and MeshCells that are repeated.

        Raises
        ------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("find_repeats not implemented")
