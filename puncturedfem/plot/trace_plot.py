"""
trace_plot.py
=============

Module containing the TracePlot class for plotting traces of functions on the
boundary of a MeshCell.
"""

import numpy as np
import matplotlib.pyplot as plt

from ..mesh.cell import MeshCell
from ..mesh.quad import Quad

PI_CHAR = r"$\pi$"


class TracePlot:
    """
    Class for plotting traces of functions on the boundary of a MeshCell.
    """

    fig_handle: plt.Figure
    t: np.ndarray
    x_ticks: np.ndarray
    x_labels: list[str]
    num_pts: int

    def __init__(
        self,
        traces: np.ndarray | list[np.ndarray],
        K: MeshCell,
        quad_dict: dict[str, Quad],
        fmt: str | list[str] = "k-",
        legend: tuple = (),
        title: str = "",
        log_scale: bool = False,
        grid: bool = True,
        filename: str = "",
        show: bool = True,
    ) -> None:
        """
        Constructor for TracePlot class.

        Parameters
        ----------
        traces : numpy.ndarray or list of numpy.ndarrays
            The traces to be plotted. Each trace must be a numpy array of the
            same length as the number of sampled boundary points of the MeshCell
        K : MeshCell
            The MeshCell whose boundary the traces are sampled on
        quad_dict : dict[str, Quad]
            A dictionary of the quadrature rules used to sample the boundary of
            the MeshCell
        fmt : str or list of str, optional
            The format string(s) used to plot the traces. If a single string is
            given, it is used for all traces. If a list of strings is given that
            is the same length as traces, each string is used for the
            corresponding trace. The default is "k-", which plots the traces as
            black solid lines.
        legend : tuple of str, optional
            The legend for the plot. The default is (), which does not display a
            legend. Must be a tuple of strings of the same length as traces.
        title : str, optional
            The title for the plot. The default is "", which does not display a
            title.
        log_scale : bool, optional
            Whether or not to use a log scale on the vertical axis. The default
            is False, which uses a linear scale.
        grid : bool, optional
            Whether or not to display a grid on the plot. The default is True,
            which displays a grid.
        filename : str, optional
            The filename to save the plot to. The default is "", which does not
            save the plot.
        show : bool, optional
            Whether or not to display the plot. The default is True, which
            displays the plot.
        """
        self._set_num_pts(K)
        self._validate_traces(traces)
        self.fig_handle = plt.figure()
        self._find_t_parameter(K, quad_dict)
        self._find_ticks_and_labels(K.num_edges)
        if isinstance(traces, np.ndarray):
            self._plot_single_trace(traces, fmt, log_scale)
        elif isinstance(traces, list):
            self._plot_traces(traces, fmt, log_scale)
        self._make_legend(legend)
        self._make_title(title)
        self._make_axis_labels(grid)
        if filename:
            plt.savefig(filename)
        if show:
            plt.show()

    def save(self, filename: str) -> None:
        """
        Saves the plot to a file.

        Parameters
        ----------
        filename : str
            The filename to save the plot to.
        """
        plt.savefig(filename)

    def show(self) -> None:
        """
        Displays the plot.
        """
        plt.show()

    def _set_num_pts(self, K: MeshCell) -> None:
        self.num_pts = K.num_pts

    def _validate_traces(self, traces: np.ndarray | list[np.ndarray]) -> None:
        if isinstance(traces, np.ndarray):
            if len(traces) != self.num_pts:
                raise ValueError(
                    "traces must be a numpy array of the same length as the "
                    "trace parameter"
                )
        elif isinstance(traces, list):
            for f_trace in traces:
                if not isinstance(f_trace, np.ndarray):
                    raise TypeError("traces must be a list of numpy arrays")
                if len(f_trace) != self.num_pts:
                    raise ValueError(
                        "traces must be a list of numpy arrays of the same "
                        "length"
                    )
        else:
            raise TypeError(
                "traces must be a numpy array or a list of numpy arrays"
            )

    def _find_t_parameter(
        self, K: MeshCell, quad_dict: dict[str, Quad]
    ) -> None:
        self.t = np.zeros((K.num_pts,))
        t0 = 0.0
        idx_start = 0
        for c in K.components:
            for e in c.edges:
                self.t[idx_start : (idx_start + e.num_pts - 1)] = (
                    t0 + quad_dict[e.quad_type].t[:-1]
                )
                idx_start += e.num_pts - 1
                t0 += 2 * np.pi

    def _find_ticks_and_labels(self, num_edges: int) -> None:
        self.x_ticks = np.linspace(0, 2 * np.pi * num_edges, num_edges + 1)
        self.x_labels = [
            "0",
        ]
        for k in range(1, num_edges + 1):
            self.x_labels.append(f"{2 * k}{PI_CHAR}")

    def _plot_traces(
        self, traces: list[np.ndarray], fmt: str | list[str], log_scale: bool
    ) -> None:
        if isinstance(fmt, str):
            for f_trace in traces:
                self._plot_single_trace(f_trace, fmt, log_scale)
        elif isinstance(fmt, list):
            if len(fmt) != len(traces):
                raise ValueError(
                    "fmt must either be a string or a list of the same length "
                    "as traces"
                )
            for k, f_trace in enumerate(traces):
                self._plot_single_trace(f_trace, fmt[k], log_scale)

    def _plot_single_trace(
        self, f_trace: np.ndarray, fmt: str | list[str], log_scale: bool
    ) -> None:
        if isinstance(fmt, list):
            fmt_str = fmt[0]
        elif isinstance(fmt, str):
            fmt_str = fmt
        else:
            raise TypeError("fmt must be a string or a list of strings")
        if log_scale:
            plt.semilogy(self.t, f_trace, fmt_str)
        else:
            plt.plot(self.t, f_trace, fmt_str)

    def _make_legend(self, legend: tuple) -> None:
        if not isinstance(legend, tuple):
            raise TypeError("legend must be a tuple of strings")
        if legend:
            plt.legend(legend)

    def _make_title(self, title: str) -> None:
        if not isinstance(title, str):
            raise TypeError("title must be a string")
        if title:
            plt.title(title)

    def _make_axis_labels(self, grid: bool) -> None:
        plt.grid(grid)
        plt.xticks(ticks=self.x_ticks, labels=self.x_labels)
