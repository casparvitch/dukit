# -*- coding: utf-8 -*-
"""
Shared plotting methods.

Functions
---------
 - `dukit.shared.plot.plot_image`
 - `dukit.shared.plot.plot_image_on_ax`
 - `dukit.shared.plot.get_colormap_range`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.shared.plot.plot_image": True,
    "dukit.shared.plot.plot_image_on_ax": True,
    "dukit.shared.plot.get_colormap_range": True,
}

# ============================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colorbar
import numpy as np
import numpy.typing as npt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

# ============================================================================

from dukit.shared.misc import dukit_warn

# ============================================================================

def plot_image(
    image_data: npt.NDArray,
    title: str = "title",
    c_map: str = "viridis",
    c_range: tuple[float, float] | tuple[None, None] = (None, None),
    c_label: str = "label",
    opath: str = "",
    show_scalebar: bool = True,
    raw_pixel_size: float = float("nan"),
    applied_binning: tuple[int, int] | int = 0,
    annotate_polygons: bool = False,
    polygon_nodes: list | None = None,
    show_tick_marks: bool = False,
    polygon_patch_params: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots an image given by image_data. Only saves figure if path given.

    Arguments
    ---------
    image_data : np array, 3D
        Data that is plot.
    title : str
        Title of figure, as well as name for save files
    c_map : str
        Colormap object used to map image_data values to a color.
    c_range : 2-tuple of floats
        Range of values in image_data to map to colors
    c_label : str
        Label for colormap axis
    opath: str  = "",
        If given saves figure here.
    show_scalebar: bool = True
        Show the scalebar(s)?
    raw_pixel_size: float = float("nan")
        Pixel size from hardware
    applied_binning: tuple[int, int] | int = 0,
        Any additional binning that HAS been applied
    annotate_polygons: bool = False,
        Annotate the polygons on image?
    polygon_nodes: list | None = None,
        List of polygon nodes, see TODO for format
    show_tick_marks: bool = False
        Show tick marks on axes
    polygon_patch_params: dict | None, default = None
        Passed to mpl.patches.Polygon.

    Returns
    -------
    fig : matplotlib Figure object
    ax : matplotlib Axis object
    """

    fig, ax = plt.subplots()

    fig, ax = plot_image_on_ax(
        fig,
        ax,
        image_data,
        title,
        c_map,
        c_range,
        c_label,
        opath=opath,
        show_scalebar=show_scalebar,
        raw_pixel_size=raw_pixel_size,
        applied_binning=applied_binning,
        annotate_polygons=annotate_polygons,
        polygon_nodes=polygon_nodes,
        show_tick_marks=show_tick_marks,
        polygon_patch_params=polygon_patch_params,
    )
    return fig, ax


# ============================================================================


def plot_image_on_ax(
    fig: plt.Figure,
    ax: plt.Axes,
    image_data: npt.NDArray,
    title: str = "title",
    c_map: str = "viridis",
    c_range: tuple[float, float] | tuple[None, None] = (None, None),
    c_label: str = "label",
    show_scalebar: bool = True,
    raw_pixel_size: float = float("nan"),
    applied_binning: tuple[int, int] | int = 0,
    annotate_polygons: bool = False,
    polygon_nodes: list | None = None,
    show_tick_marks: bool = False,
    polygon_patch_params: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots an image given by image_data onto given figure and ax.

    Does not save any data.

    Arguments
    ---------
    fig : matplotlib Figure object
    ax : matplotlib Axis object
    image_data : np array, 3D
        Data that is plot.
    title : str
        Title of figure, as well as name for save files
    c_map : str
        Colormap object used to map image_data values to a color.
    c_range : 2-tuple of floats
        Range of values in image_data to map to colors
    c_label : str
        Label for colormap axis
    show_scalebar: bool = True
        Show the scalebar(s)?
    raw_pixel_size: float = float("nan")
        Pixel size from hardware
    applied_binning: tuple[int, int] | int = 0,
        Any additional binning that HAS been applied
    annotate_polygons: bool = False,
        Annotate the polygons on image?
    polygon_nodes: list | None = None,
        List of polygon nodes, see TODO for format
    show_tick_marks: bool = False
        Show tick marks on axes
    polygon_patch_params: dict | None, default = None
        Passed to mpl.patches.Polygon.

    Returns
    -------
    fig : matplotlib Figure object
    ax : matplotlib Axis object
    """

    imshowed = ax.imshow(image_data, cmap=c_map, vmin=c_range[0], vmax=c_range[1])

    ax.set_title(title)

    cbar = _add_colorbar(imshowed, fig, ax)
    cbar.ax.set_ylabel(c_label, rotation=270)

    if show_scalebar and raw_pixel_size:
        if isinstance(applied_binning, tuple):
            bin_x, bin_y = applied_binning
            pixel_x = raw_pixel_size * bin_x if bin_x else raw_pixel_size
            pixel_y = raw_pixel_size * bin_y if bin_y else raw_pixel_size
            scalebar_x = ScaleBar(pixel_x)
            scalebar_y = ScaleBar(pixel_y, rotation="vertical")
            ax.add_artist(scalebar_x)
            ax.add_artist(scalebar_y)
        else:
            pixel = (
                raw_pixel_size * applied_binning if applied_binning else raw_pixel_size
            )
            scalebar = ScaleBar(pixel)
            ax.add_artist(scalebar)

    if polygon_nodes and annotate_polygons:
        if polygon_patch_params is None:
            polygon_patch_params = {
                "facecolor": None,
                "edgecolor": "k",
                "linestyle": "dashed",
                "fill": False,
            }
        for poly in polygon_nodes:
            # polygons reversed to (x,y) indexing for patch
            ax.add_patch(
                matplotlib.patches.Polygon(
                    np.dstack((poly[:, 1], poly[:, 0]))[0],
                    **polygon_patch_params,
                )
            )

    if not show_tick_marks:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    return fig, ax


# ============================================================================


def _add_colorbar(
    imshowed: mpl.cm.ScalarMappable,
    fig: plt.Figure,
    ax: plt.Axes,
    aspect: float = 20.0,
    pad_fraction: float = 1.0,
    **kwargs
) -> matplotlib.colorbar.Colorbar:
    """
    Adds a colorbar to matplotlib axis

    Arguments
    ---------
    imshowed : image as returned by ax.imshow
    fig : matplotlib Figure object
    ax : matplotlib Axis object

    Returns
    -------
    cbar : matplotlib colorbar object

    Optional Arguments
    ------------------
    aspect : int
        Reciprocal of aspect ratio passed to new colorbar axis width. Default: 20.
    pad_fraction : int
        Fraction of new colorbar axis width to pad from image. Default: 1.

    **kwargs : other keyword arguments
        Passed to fig.colorbar.

    """
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1.0 / aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cbar = fig.colorbar(imshowed, cax=cax, **kwargs)
    tick_locator = mpl.ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.linewidth = 0.5  # type: ignore
    return cbar


# ============================================================================


def get_colormap_range(
    c_range_type: str,
    c_range_vals: tuple[float, ...],
    image: npt.NDArray,
    auto_sym_zero: bool = True,
) -> tuple[float, float]:
    """
    Produce a colormap range to plot image from, using the options in c_range_dict.

    Arguments
    ---------
    c_range_type : str
         - "min_max" : map between minimum and maximum values in image.
         - "deviation_from_mean" : requires c_range_vals be a 1-tuple float
           between 0 and 1 'dev'. Maps between (1 - dev) * mean and (1 + dev) * mean.
         - "min_max_symmetric_about_mean" : map symmetrically about mean, capturing all
            values in image (default).
         - "min_max_symmetric_about_zero" : map symmetrically about zero, capturing all
            values in image.
         - "percentile" : requires c_range_vals be a 2-tuple w vals between 0 and 100.
            Maps the range between those percentiles of the data.
         - "percentile_symmetric_about_zero" : requires c_range_vals be a 2-tuple
            w vals between 0 and 100. Maps symmetrically about zero, capturing
            all values between those percentiles in the data (plus perhaps a bit more to
            ensure symmety)
         - "strict_range" : requiresc_range_vals 2-tuple.
            Maps colors directly between the values given.
         - "mean_plus_minus" : mean plus or minus this value.
            c_range_vals must be a 1-tuple float
    c_range_vals : tuple
        See above for allowed options
    auto_sym_zero : bool, default=True
        Try and make symmetric around zero, if logical?

    image : array-like
        2D array (image) that fn returns colormap range for.

    Returns
    -------
    c_range : list length 2
        i.e. [min value to map to a color, max value to map to a color]
    """

    # mostly these are just checking that the input values are valid

    warning_messages = {
        "deviation_from_mean": """Invalid c_range_dict['values'] encountered.
        For c_range type 'deviation_from_mean', c_range_dict['values'] must be a float,
        between 0 and 1. Changing to 'min_max_symmetric_about_mean' c_range.""",
        "strict_range": """Invalid c_range_dict['values'] encountered.
        For c_range type 'strict_range', c_range_dict['values'] must be a a list of 
        length 2, with elements that are floats or ints.
        Changing to 'min_max_symmetric_about_mean' c_range.""",
        "mean_plus_minus": """Invalid c_range_dict['values'] encountered.
        For c_range type 'mean_plus_minus', c_range_dict['values'] must be an int or 
        float. Changing to 'min_max_symmetric_about_mean' c_range.""",
        "percentile": """Invalid c_range_dict['values'] encountered.
        For c_range type 'percentile', c_range_dict['values'] must be a list of length 2,
         with elements (preferably ints) in [0, 100].
         Changing to 'min_max_symmetric_about_mean' c_range.""",
        "percentile_symmetric_about_zero": """Invalid c_range_dict['values'] encountered.
        For c_range type 'percentile', c_range_dict['values'] must be a list of length 2,
         with elements (preferably ints) in [0, 100].
         Changing to 'min_max_symmetric_about_mean' c_range.""",
    }
    if not c_range_type:
        c_range_type = "min_max_symmetric_about_mean"
        c_range_vals = ()

    if c_range_type not in ["percentile", "min_max"]:
        auto_sym_zero = False

    if auto_sym_zero and np.any(image < 0) and np.any(image > 0):
        if c_range_type == "min_max":
            c_range_type = "min_max_symmetric_about_zero"
        elif c_range_type == "percentile":
            c_range_type = "percentile_symmetric_about_zero"

    range_calculator_dict = {
        "min_max": _min_max,
        "deviation_from_mean": _deviation_from_mean,
        "min_max_symmetric_about_mean": _min_max_sym_mean,
        "min_max_symmetric_about_zero": _min_max_sym_zero,
        "percentile": _percentile,
        "percentile_symmetric_about_zero": _percentile_sym_zero,
        "strict_range": _strict_range,
        "mean_plus_minus": _mean_plus_minus,
    }

    if c_range_type == "strict_range":
        if (
            not isinstance(c_range_vals, (list, tuple))
            or len(c_range_vals) != 2  # noqa: W503
            or (not isinstance(c_range_vals[0], (float, int)))  # noqa: W503
            or (not isinstance(c_range_vals[1], (float, int)))  # noqa: W503
            or c_range_vals[0] > c_range_vals[1]  # noqa: W503
        ):
            dukit_warn(warning_messages[c_range_type])
            return _min_max_sym_mean(image, ())
    elif c_range_type == "mean_plus_minus":
        if not isinstance(c_range_vals[0], float):
            dukit_warn(warning_messages[c_range_type])
            return _min_max_sym_mean(image, ())
    elif c_range_type == "deviation_from_mean":
        if (
            not isinstance(c_range_vals, (float, int))
            or c_range_vals < 0  # noqa: W503
            or c_range_vals > 1  # noqa: W503
        ):
            dukit_warn(warning_messages[c_range_type])
            return _min_max_sym_mean(image, ())

    elif c_range_type.startswith("percentile"):
        if (
            not isinstance(c_range_vals, tuple)
            or len(c_range_vals) != 2  # noqa: W503
            or not isinstance(c_range_vals[0], float)
            or not isinstance(c_range_vals[1], float)
            or not 100 >= c_range_vals[0] >= 0
            or not 100 >= c_range_vals[1] >= 0
        ):
            dukit_warn(warning_messages[c_range_type])
            return _min_max_sym_mean(image, ())
    return range_calculator_dict[c_range_type](image, c_range_vals)


# ============================


def _min_max(
    image: npt.NDArray, c_range_values: tuple[float, ...]
) -> tuple[float, float]:
    """
    Map between minimum and maximum values in image

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `qdmpy.plot.common.get_colormap_range`
    """
    return np.nanmin(image), np.nanmax(image)


def _strict_range(
    image: npt.NDArray, c_range_values: tuple[float, ...]
) -> tuple[float, float]:
    """
    Map between c_range_values

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `qdmpy.plot.common.get_colormap_range`
    """
    return c_range_values[0], c_range_values[1]


def _min_max_sym_mean(
    image: npt.NDArray, c_range_values: tuple[float, ...]
) -> tuple[float, float]:
    """
    Map symmetrically about mean, capturing all values in image.

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `qdmpy.plot.common.get_colormap_range`
    """
    minimum = np.nanmin(image)
    maximum = np.nanmax(image)
    mean = np.mean(image)
    max_distance_from_mean = np.max([abs(maximum - mean), abs(minimum - mean)])
    return mean - max_distance_from_mean, mean + max_distance_from_mean


def _min_max_sym_zero(
    image: npt.NDArray, c_range_values: tuple[float, ...]
) -> tuple[float, float]:
    """
    Map symmetrically about zero, capturing all values in image.

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `qdmpy.plot.common.get_colormap_range`
    """
    min_abs = np.abs(np.nanmin(image))
    max_abs = np.abs(np.nanmax(image))
    larger = np.nanmax([min_abs, max_abs])
    return -larger, larger


def _deviation_from_mean(
    image: npt.NDArray, c_range_values: tuple[float, ...]
) -> tuple[float, float]:
    """
    Map a (decimal) deviation from mean,
    i.e. between (1 - dev) * mean and (1 + dev) * mean

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `qdmpy.plot.common.get_colormap_range`
    """
    return (1 - c_range_values[0]) * np.mean(image), (1 + c_range_values[0]) * np.mean(
        image
    )


def _percentile(
    image: npt.NDArray, c_range_values: tuple[float, ...]
) -> tuple[float, float]:
    """
    Maps the range between two percentiles of the data.

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `qdmpy.plot.common.get_colormap_range`
    """
    return tuple(np.nanpercentile(image, c_range_values))


def _percentile_sym_zero(
    image: npt.NDArray, c_range_values: tuple[float, ...]
) -> tuple[float, float]:
    """
    Maps the range between two percentiles of the data, but ensuring symmetry about zero

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `qdmpy.plot.common.get_colormap_range`
    """
    plow, phigh = np.nanpercentile(image, c_range_values)  # e.g. [10, 90]
    val = max(abs(plow), abs(phigh))
    return -val, val


def _mean_plus_minus(
    image: npt.NDArray, c_range_values: tuple[float, ...]
) -> tuple[float, float]:
    """
    Maps the range to mean +- value given in c_range_values

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `qdmpy.plot.common.get_colormap_range`
    """
    mean = np.mean(image)
    return mean - c_range_values, mean + c_range_values


# ============================================================================
