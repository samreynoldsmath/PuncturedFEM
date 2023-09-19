from .cell import cell
from .edge import edge


class planar_mesh:
    """Handles planar mesh geometry and topology"""

    num_edges: int
    num_cells: int
    num_verts: int
    edges: list[edge]
    vert_idx_list: list[int]
    cell_idx_list: list[int]

    def __init__(self, edges: list[edge], verbose=True) -> None:
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
        s = "planar_mesh:"
        s += "\n\tnum_verts: %d" % self.num_verts
        s += "\n\tnum_edges: %d" % self.num_edges
        s += "\n\tnum_cells: %d" % self.num_cells
        return s

    # EDGES ##################################################################
    def compute_num_edges(self) -> None:
        self.num_edges = len(self.edges)

    def add_edge(self, e: edge) -> None:
        if not isinstance(e, edge):
            raise TypeError("e must be an edge")
        if e in self.edges:
            return
        self.edges.append(e)
        self.num_edges += 1

    def add_edges(self, edges: list[edge]) -> None:
        for e in edges:
            self.add_edge(e)

    def set_edge_ids(self) -> None:
        for k, e in enumerate(self.edges):
            e.set_id(k)

    # VERTICES ###############################################################

    def compute_num_verts(self) -> None:
        self.num_verts = len(self.vert_idx_list)

    def build_vert_idx_list(self) -> None:
        self.vert_idx_list = []
        for e in self.edges:
            if not e.is_loop:
                for vert_idx in [e.anchor.id, e.endpnt.id]:
                    if vert_idx >= 0 and vert_idx not in self.vert_idx_list:
                        self.vert_idx_list.append(vert_idx)
        self.vert_idx_list.sort()
        self.compute_num_verts()

    # CELLS ##################################################################

    def compute_num_cells(self) -> None:
        self.num_cells = len(self.cell_idx_list)

    def build_cell_idx_list(self) -> None:
        self.cell_idx_list = []
        for e in self.edges:
            for cell_idx in [e.pos_cell_idx, e.neg_cell_idx]:
                if cell_idx >= 0 and cell_idx not in self.cell_idx_list:
                    self.cell_idx_list.append(cell_idx)
        self.cell_idx_list.sort()
        self.compute_num_cells()

    def get_cell(self, cell_idx: int) -> cell:
        if cell_idx not in self.cell_idx_list:
            raise Exception("cell_idx is not in cell_idx_list")
        edges = []
        for e in self.edges:
            if e.pos_cell_idx == cell_idx or e.neg_cell_idx == cell_idx:
                edges.append(e)
        return cell(id=cell_idx, edges=edges)

    def get_abs_cell_idx(self, cell_idx: int) -> int:
        return self.cell_idx_list.index(cell_idx)

    # BOUNDARY IDENTIFICATION ################################################

    def vert_is_on_boundary(self, vert_idx: int) -> bool:
        for e in self.edges:
            if e.anchor.id == vert_idx or e.endpnt.id == vert_idx:
                if e.pos_cell_idx < 0 or e.neg_cell_idx < 0:
                    return True
        return False

    def edge_is_on_boundary(self, edge_idx: int) -> bool:
        e = self.edges[edge_idx]
        return e.pos_cell_idx < 0 or e.neg_cell_idx < 0

    # REPEAT DETECTION #######################################################

    # TODO find edges and cells that are repeated

    def find_repeats(self) -> None:
        pass
        # raise NotImplementedError("find_repeats not implemented")
