"""
trace_plot.py
=============

Module containing the TracePlot class for plotting traces of functions on the
boundary of a MeshCell.
"""

from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from ..locfun.trace import DirichletTrace
from ..mesh.cell import MeshCell
from ..mesh.quad import Quad
from .plot_util import save_figure

PI_CHAR = r"$\pi$"

TraceLike = Union[np.ndarray, DirichletTrace]


class TracePlot:
    """
    Class for plotting traces of functions on the boundary of a MeshCell.
    """

    fig_handle: plt.Figure
    t: np.ndarray
    traces: list[np.ndarray]
    x_ticks: np.ndarray
    x_labels: list[str]
    num_pts: int
    fmt: str | list[str]
    legend: tuple
    title: str
    log_scale: bool
    show_grid: bool

    def __init__(
        self,
        traces: Union[TraceLike, list[TraceLike]],
        K: MeshCell,
        quad_dict: dict[str, Quad],
        fmt: str | list[str] = "k-",
        legend: tuple = (),
        title: str = "",
        log_scale: bool = False,
        show_grid: bool = True,
        max_num_ticks: int = 9,
    ) -> None:
        """
        Constructor for TracePlot class.

        Parameters
        ----------
        traces : Union[TraceLike, list[TraceLike]]
            The traces to be plotted. Each trace must be a DirichletTrace or a
            numpy.ndarray of the same length as the number of sampled boundary
            points of the MeshCell.
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
        show_grid : bool, optional
            Whether or not to display a grid on the plot. The default is True,
            which displays a grid.
        max_num_ticks : int, optional
            The maximum number of ticks to display on the x-axis. The default is
            9.
        """
        self._set_num_pts(K)
        self._find_t_parameter(K, quad_dict)
        self._find_ticks_and_labels(K.num_edges, max_num_ticks)
        self.set_traces(traces)
        self.set_format(fmt)
        self.set_legend(legend)
        self.set_title(title)
        self.set_log_scale(log_scale)
        self.set_grid(show_grid)

    def draw(self, show_plot: bool = True, filename: str = "") -> None:
        """
        Creates the plot.

        Parameters
        ----------
        show_plot : bool, optional
            Whether to show the plot. The default is True.
        filename : str, optional
            The filename to save the plot to. The default is "", which results
            in no file being saved.
        """
        self.fig_handle = plt.figure()
        self._plot_traces()
        self._make_legend(self.legend)
        self._make_title(self.title)
        self._make_axis_labels()
        plt.grid(self.show_grid)
        if filename:
            save_figure(filename)
        if show_plot:
            plt.show()
        plt.close()

    def set_traces(self, traces: Union[TraceLike, list[TraceLike]]) -> None:
        """
        Sets the traces to be plotted.
        """
        if not isinstance(traces, list):
            traces = [traces]
        self.traces = []
        for f in traces:
            if isinstance(f, DirichletTrace):
                self.traces.append(f.values)
                f = f.values
            if not isinstance(f, np.ndarray):
                raise TypeError(
                    "Each trace must be a DirichletTrace or a numpy.ndarray"
                )
            if len(f) != self.num_pts:
                raise ValueError(
                    "Each trace must be a numpy array of the same length as "
                    "the number of sampled boundary points of the MeshCell"
                )
            self.traces.append(f)

    def set_format(self, fmt: str | list[str]) -> None:
        """
        Sets the format string(s) used to plot the traces.
        """
        self._validate_format(fmt)
        self.fmt = fmt

    def set_legend(self, legend: tuple) -> None:
        """
        Sets the legend for the plot.
        """
        self._validate_legend(legend)
        self.legend = legend

    def set_title(self, title: str) -> None:
        """
        Sets the title for the plot.
        """
        if not isinstance(title, str):
            raise TypeError("title must be a string")
        self.title = title

    def set_grid(self, show_grid: bool) -> None:
        """
        Sets whether or not to display a grid on the plot.
        """
        if not isinstance(show_grid, bool):
            raise TypeError("show_grid must be a boolean")
        self.show_grid = show_grid

    def set_log_scale(self, log_scale: bool) -> None:
        """
        Sets whether or not to use a log scale on the vertical axis.
        """
        if not isinstance(log_scale, bool):
            raise TypeError("log_scale must be a boolean")
        self.log_scale = log_scale

    def _set_num_pts(self, K: MeshCell) -> None:
        self.num_pts = K.num_pts

    def _validate_legend(self, legend: tuple) -> None:
        if not isinstance(legend, tuple):
            raise TypeError("legend must be a tuple of strings")
        if legend:
            if len(legend) != len(self.traces):
                raise ValueError(
                    "legend must be a tuple of strings of the same length as "
                    "traces"
                )

    def _validate_format(self, fmt: str | list[str]) -> None:
        if isinstance(fmt, str):
            pass
        elif isinstance(fmt, list):
            if len(fmt) != len(self.traces):
                raise ValueError(
                    "fmt must either be a string or a list of the same length "
                    "as traces"
                )
        else:
            raise TypeError("fmt must be a string or a list of strings")

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

    def _find_ticks_and_labels(
        self, num_edges: int, max_num_ticks: int
    ) -> None:
        interval_length = 2 * np.pi * num_edges
        num_ticks = min(num_edges + 1, max_num_ticks)
        rotations_per_tick = num_edges // (num_ticks - 1)
        self.x_labels = ["0"]
        if num_edges % (num_ticks - 1) == 0:
            self._find_ticks_and_labels_no_remainder(
                interval_length, rotations_per_tick, num_ticks
            )
        else:
            self._find_ticks_and_labels_with_remainder(
                interval_length, rotations_per_tick, num_ticks, num_edges
            )

    def _find_ticks_and_labels_no_remainder(
        self, interval_length: float, rotations_per_tick: int, num_ticks: int
    ) -> None:
        self.x_ticks = np.linspace(0, interval_length, num_ticks)
        self.x_labels += [
            f"{2 * k * rotations_per_tick}{PI_CHAR}"
            for k in range(1, num_ticks)
        ]

    def _find_ticks_and_labels_with_remainder(
        self,
        interval_length: float,
        rotations_per_tick: int,
        num_ticks: int,
        num_edges: int,
    ) -> None:
        self.x_ticks = np.zeros(num_ticks)
        self.x_ticks[:-1] = np.linspace(
            0,
            2 * np.pi * rotations_per_tick * (num_ticks - 1),
            num_ticks - 1,
        )
        self.x_ticks[-1] = interval_length
        self.x_labels += [
            f"{2 * k * rotations_per_tick}{PI_CHAR}"
            for k in range(1, num_ticks - 1)
        ] + [f"{2 * num_edges}{PI_CHAR}"]

    def _plot_traces(self) -> None:
        if isinstance(self.traces, np.ndarray):
            if isinstance(self.fmt, str):
                fmt_str = self.fmt
            elif isinstance(self.fmt, list):
                fmt_str = self.fmt[0]
            self._plot_single_trace(self.traces, fmt_str)
        elif isinstance(self.traces, list):
            if isinstance(self.fmt, str):
                for f_trace in self.traces:
                    self._plot_single_trace(f_trace, self.fmt)
            elif isinstance(self.fmt, list):
                for f_trace, fmt in zip(self.traces, self.fmt):
                    self._plot_single_trace(f_trace, fmt)

    def _plot_single_trace(self, vals: np.ndarray, fmt: str) -> None:
        if self.log_scale:
            plt.semilogy(self.t, vals, fmt)
        else:
            plt.plot(self.t, vals, fmt)

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

    def _make_axis_labels(self) -> None:
        plt.xticks(ticks=self.x_ticks, labels=self.x_labels)
