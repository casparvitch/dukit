# -*- coding: utf-8 -*-
"""Always lowest in import heirarchy

dukit_warn, mpl_set_run_config, crop_roi, crop_sweep, \
    rebin_image_stack, smooth_image_stack, sum_spatially
Functions
---------
 - `dukit.shared.dukit_warn`
 - `dukit.shared.mpl_set_run_config`
 - `dukit.shared.crop_roi`
 - `dukit.shared.crop_sweep`
 - `dukit.shared.rebin_image_stack`
 - `dukit.shared.smooth_image_stack`
 - `dukit.shared.sum_spatially`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.shared.dukit_warn": True,
    "dukit.shared.mpl_set_run_config": True,
    "dukit.shared.crop_roi": True,
    "dukit.shared.crop_sweep": True,
    "dukit.shared.rebin_image_stack": True,
    "dukit.shared.smooth_image_stack": True,
    "dukit.shared.sum_spatially": True,
}

# ============================================================================

import warnings
import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter
import matplotlib as mpl

# ============================================================================

from dukit.shared.rebin import rebin

# ============================================================================

DEFAULT_RCPARAMS = {
    "figure.constrained_layout.use": False,
    "figure.figsize": [6.4, 4.8],
    "figure.dpi": 80,
    "figure.max_open_warning": 30,
    "lines.linewidth": 0.8,
    "lines.markersize": 3,
    "xtick.labelsize": 10,
    "xtick.major.size": 4,
    "xtick.direction": "in",
    "ytick.labelsize": 10,
    "ytick.direction": "in",
    "ytick.major.size": 4,
    "legend.fontsize": "small",
    "legend.loc": "lower left",
    "scalebar.location": "lower right",
    "scalebar.width_fraction": 0.015,
    "scalebar.box_alpha": 0.5,
    "scalebar.scale_loc": "top",
    "scalebar.sep": 1,
    "axes.formatter.useoffset": False,
    "axes.formatter.use_mathtext": True,
    "errorbar.capsize": 3.0
}
"""Set of rcparams we've casually decided are reasonable."""


# ============================================================================


def dukit_warn(msg: str):
    """Throw a custom _DUKITWarning with message 'msg'."""
    warnings.warn(msg, _DUKITWarning)


class _DUKITWarning(Warning):
    """allows us to separate dukit warnings from those in other packages."""


# ============================================================================

def mpl_set_run_config(default: bool = True, **kwargs):
    """
    Set the matplotlib runtime configuration (rcparams).

    Parameters
    ----------
    default : bool
        Use dukit default rcparams?
    kwargs
        kwargs to pass to rcparams, see
        https://matplotlib.org/stable/users/explain/customizing.html
    """
    if default:
        mpl.rcParams.update(DEFAULT_RCPARAMS)
    else:
        mpl.rcParams.update(**kwargs)


# ============================================================================


def crop_roi(
        seq: npt.ArrayLike, roi_coords: tuple[int, int, int, int]
) -> npt.NDArray:
    """

    Parameters
    ----------
    seq : array-like
        Image or image-stack you want to crop
    roi_coords: 4-tuple
        start_x: int, start_y: int, end_x: int, end_y: int
        If any are -1, then it sets to the edge of image.

    Returns
    -------
    seq_cropped : np.ndarray
        Image or image-stack cropped.
    """
    seq = np.asarray(seq)
    roi = _define_roi(seq, roi_coords)
    if len(np.shape(seq)) == 2:
        return seq[roi[0], roi[1]].copy()
    return seq[:, roi[0], roi[1]]


def _define_roi(
        img: npt.ArrayLike, roi_coords: tuple[int, int, int, int]
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Returns
    -------
    roi : 2-tuple of ndarrays
        Meshgrid that can be used to index into arrays,
        e.g. sig[sweep_param, roi[0], roi[1]]
    """
    try:
        size_h, size_w = np.shape(img)[1:]
    except ValueError:  # not enough values to unpack -> 2d image not 3d
        size_h, size_w = np.shape(img)
    start_x, start_y, end_x, end_y = _check_start_end_rectangle(
            *roi_coords, size_w, size_h
    )
    return _define_area_roi(start_x, start_y, end_x, end_y)


# ============================================================================


def _define_area_roi(
        start_x: int, start_y: int, end_x: int, end_y: int
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Returns
    -------
    roi : 2-tuple of ndarrays
        Meshgrid that can be used to index into arrays,
        e.g. sig[sweep_param, roi[0], roi[1]]
    """
    x: npt.NDArray = np.linspace(start_x, end_x, end_x - start_x + 1, dtype=int)
    y: npt.NDArray = np.linspace(start_y, end_y, end_y - start_y + 1, dtype=int)
    xv, yv = np.meshgrid(x, y)
    return yv, xv


# ============================================================================


def _check_start_end_rectangle(
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        full_size_w: int,
        full_size_h: int,
) -> tuple[int, int, int, int]:
    """Restricts roi params to be within image dimensions."""
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    if end_x < 0:
        end_x = full_size_w - 1
    if end_y < 0:
        end_y = full_size_h - 1
    if start_x >= end_x:
        dukit_warn(
                f"Rectangle ends [{end_x}] before it starts [{start_x}] (in x), "
                + "swapping them"
        )
        start_x, end_x = end_x, start_x
    if start_y >= end_y:
        dukit_warn(
                f"Rectangle ends [{end_y}] before it starts [{start_y}] (in y), "
                + "swapping them"
        )
        start_y, end_y = end_y, start_y
    if start_x >= full_size_w:
        dukit_warn(
                f"Rectangle starts [{start_x}] outside image [{full_size_w}] "
                + "(too large in x), setting to zero."
        )
        start_x = 0

    if start_y >= full_size_h:
        dukit_warn(
                f"Rectangle starts outside [{start_y}] image [{full_size_h}] "
                + "(too large in y), setting to zero."
        )
        start_y = 0

    if end_x >= full_size_w:
        dukit_warn(
                f"Rectangle too big in x [{end_x}], cropping to image [{full_size_w}].\n"
        )
        end_x = full_size_w - 1
    if end_y >= full_size_h:
        dukit_warn(
                f"Rectangle too big in y [{end_y}], cropping to image [{full_size_h}].\n"
        )
        end_y = full_size_h - 1

    return start_x, start_y, end_x, end_y


# ============================================================================


def crop_sweep(
        sweep_arr: npt.NDArray,
        sig: npt.NDArray,
        ref: npt.NDArray,
        sig_norm: npt.NDArray,
        rem_start=1,
        rem_end=0,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Crop spectral dimension. Usually used to remove first one/few points of e.g. ODMR.

    Parameters
    ----------
    sweep_arr, sig, ref, sig_norm : npdarray's
        ndarrays to be cropped in spectral dimension. (sweep_arr only has spec dim).
    rem_start, rem_end : int
        How many pts to remove from start & end of spectral dimension.

    Returns
    -------
    sweep_arr, sig, ref, sig_norm : npdarray's
        All as input, put with spectral dimension cropped to arr[rem_start:-rem_end]
    """
    end = -rem_end if rem_end > 0 else None

    if rem_start < 0:
        dukit_warn("rem_start must be >=0, setting to zero now.")
        rem_start = 0
    if rem_end < 0:
        dukit_warn("rem_end must be >=0, setting to zero now.")
        rem_end = 0

    return (
        sweep_arr[rem_start:end].copy(),
        sig[rem_start:end].copy(),
        ref[rem_start:end].copy(),
        sig_norm[rem_start:end].copy(),
    )


# ============================================================================


def smooth_image_stack(
        stack: npt.NDArray, sigma: tuple[float, float] | float, truncate: float = 4.0
) -> npt.NDArray:
    """
    Smooth image stack in spatial dimensions with gaussian.

    Parameters
    ----------
    stack : ndarray
    sigma : 2-tuple of floats or float
        If float, then symmetric smoothing in each dim, otherwise tuple(x, y)
    truncate : float, default=4.0
        Truncate the filter at this many standard deviations.
        See scipy.ndimage.gaussian_filter.

    Returns
    -------
    smooth_stack : ndarray
        Image stack smoothed in spatial dimensions.
    """
    if isinstance(sigma, tuple):
        return gaussian_filter(stack, sigma=(0, sigma[1], sigma[0]), truncate=truncate)
    return gaussian_filter(stack, sigma=(0, sigma, sigma), truncate=truncate)


# ============================================================================


def rebin_image_stack(
        stack: npt.NDArray, additional_bins: tuple[int, int] | int
) -> npt.NDArray:
    """
    Rebin image stack in spatial dimensions.

    Parameters
    ----------
    stack : ndarray
    additional_bins : 2-tuple of ints, or int
        Binning in-plane.  Make it a power of 2.
        Binning in x then y if 2-tuple, else symmetric.

    Returns
    -------
    smooth_stack : ndarray
        Image stack smoothed in spatial dimensions.
    """
    if not additional_bins:
        return stack
    if isinstance(additional_bins, tuple):
        return rebin(
                stack, factor=(1, additional_bins[1], additional_bins[0]), func=np.mean
        )
    return rebin(
            stack,
            factor=(1, additional_bins, additional_bins),
            func=np.mean,
    )


# ============================================================================
def sum_spatially(seq: npt.ArrayLike) -> npt.NDArray:
    """Sum over 0th (spectral) dim of seq if 3D, else return as is."""
    seq = np.asarray(seq)
    if len(np.shape(seq)) == 3:
        return np.sum(seq, axis=0)
    return seq
