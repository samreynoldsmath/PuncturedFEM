"""
GlobalKey.py
=============

Module containing the GlobalKey class, which is used to represent a global key,
which is used to index the global function space.
"""


class GlobalKey:
    """
    Represents a global key, which is used to index the global function space.
    """

    fun_type: str
    vert_idx: int
    edge_idx: int
    edge_space_idx: int
    bubb_space_idx: int
    glob_idx: int
    is_on_boundary: bool

    def __init__(
        self,
        fun_type: str,
        edge_idx: int = -1,
        vert_idx: int = -1,
        bubb_space_idx: int = -1,
        edge_space_idx: int = -1,
    ) -> None:
        """
        Constructor for GlobalKey class.

        Parameters
        ----------
        fun_type : str
            Type of function, either 'Vert', 'Edge', or 'bubb'
        edge_idx : int, optional
            Index of Edge, by default -1
        vert_idx : int, optional
            Index of vertex, by default -1
        bubb_space_idx : int, optional
            Index of bubble space, by default -1
        edge_space_idx : int, optional
            Index of Edge space, by default -1
        """
        self.set_fun_type(fun_type)
        self.set_vert_idx(vert_idx)
        self.set_edge_idx(edge_idx)
        self.set_edge_space_idx(edge_space_idx)
        self.set_bubb_space_idx(bubb_space_idx)

    def set_fun_type(self, fun_type: str) -> None:
        """
        Set the function type, either 'Vert', 'Edge', or 'bubb'.
        """
        if fun_type not in ["Vert", "Edge", "bubb"]:
            raise ValueError("fun_type must be Vert, Edge, or bubb")
        self.fun_type = fun_type

    def set_vert_idx(self, vert_idx: int) -> None:
        """
        Set the vertex index.
        """
        if not isinstance(vert_idx, int):
            raise TypeError("vert_idx must be an integer")
        if not self.fun_type == "Vert":
            self.vert_idx = -1
        else:
            self.vert_idx = vert_idx

    def set_edge_idx(self, edge_idx: int) -> None:
        """
        Set the Edge index.
        """
        if not isinstance(edge_idx, int):
            raise TypeError("edge_idx must be an integer")
        if not self.fun_type == "Edge":
            self.edge_idx = -1
        else:
            self.edge_idx = edge_idx

    def set_edge_space_idx(self, edge_space_idx: int) -> None:
        """
        Set the Edge space index.
        """
        if not isinstance(edge_space_idx, int):
            raise TypeError("edge_space_idx must be an integer")
        if not self.fun_type == "Edge":
            self.edge_space_idx = -1
        else:
            self.edge_space_idx = edge_space_idx

    def set_bubb_space_idx(self, bubb_space_idx: int) -> None:
        """
        Set the bubble space index.
        """
        if not isinstance(bubb_space_idx, int):
            raise TypeError("bubb_space_idx must be an integer")
        if not self.fun_type == "bubb":
            self.bubb_space_idx = -1
        else:
            self.bubb_space_idx = bubb_space_idx

    def set_glob_idx(self, glob_idx: int) -> None:
        """
        Set the global index.
        """
        if not isinstance(glob_idx, int):
            raise TypeError("glob_idx must be an integer")
        if glob_idx < 0:
            raise ValueError("glob_idx must be nonnegative")
        self.glob_idx = glob_idx
