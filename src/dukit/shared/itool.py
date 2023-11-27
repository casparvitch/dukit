# -*- coding: utf-8 -*-
"""
This module holds misc image tooling.

Functions
---------
 - `dukit.shared.itool.mask_polygons`
 - `dukit.shared.itool.get_im_filtered`
 - `dukit.shared.itool.get_background`
 - `dukit.shared.itool.mu_sigma_inside_polygons`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.shared.itool.mask_polygons": True,
    "dukit.shared.itool.get_im_filtered": True,
    "dukit.shared.itool.get_background": True,
    "dukit.shared.itool.mu_sigma_inside_polygons": True,
}

# ============================================================================

import numpy as np
import numpy.typing as npt
from numpy.polynomial.polynomial import polyval2d
from scipy.optimize import least_squares
from scipy.interpolate import griddata
import scipy.ndimage
from typing import Callable

# ============================================================================

import dukit.shared.polygon
from dukit.shared.misc import dukit_warn

# ============================================================================


def mask_polygons(
    image: npt.NDArray, polygons: list | None = None, invert_mask: bool = False
) -> npt.NDArray:
    """Mask image for the given polygon regions.

    Arguments
    ---------
    image : 2D array-like
        Image array to mask.
    polygons : list, optional
        List of `qdmpy.shared.polygon.Polygon` objects.
        (the default is None, where image is returned with no mask)
    invert_mask : bool, optional
        Invert mask such that background is masked, not polygons (i.e. polygons
        will be operated on if array is passed to np.mean instead of background).
        (the default is False)

    Returns
    -------
    masked_im : np.ma.MaskedArray
        image, now masked
    """
    image = np.array(image)
    if polygons is None:
        return np.ma.masked_array(image, mask=np.zeros(image.shape))
    if len(image.shape) != 2:
        raise ValueError("image is not a 2D array")

    if not isinstance(polygons, (list, tuple)) or not isinstance(
        polygons[0], dukit.shared.polygon.Polygon
    ):
        raise TypeError("polygons were not None, a list or a list of Polygon objects")

    ylen, xlen = image.shape
    masked_area = np.full(image.shape, True)  # all masked to start with

    grid_y, grid_x = np.meshgrid(range(ylen), range(xlen), indexing="ij")

    for p in polygons:
        in_or_out = p.is_inside(grid_y, grid_x)
        # mask all vals that are not background
        m = np.ma.masked_greater_equal(in_or_out, 0).mask
        masked_area = np.logical_and(masked_area, ~m)

    msk = np.logical_not(masked_area) if not invert_mask else masked_area

    return np.ma.masked_array(image, mask=msk)


def get_background(
    image: npt.NDArray,
    method: str,
    polygons: list | None = None,
    sigma_clip: bool = False,
    sigma_clip_sigma: float = 3,
    no_bground_if_clip_fails: bool = False,
    **method_params_dict,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Returns a background for given image, via chosen method.

    Methods available:

    - "fix_zero"
        - Fix background to be a constant offset (z value)
        - params required in method_params_dict:
            "zero" an int/float, defining the constant offset of the background
    - "three_point"
        - Calculate plane background with linear algebra from three [x,y]
          lateral positions given
        - params required in method_params_dict:
            - "points" a len-3 iterable containing [x, y] points
    - "mean"
        - background calculated from mean of image
        - no params required
    - "poly"
        - background calculated from polynomial fit to image.
        - params required in method_params_dict:
            - "order": an int, the 'order' polynomial to fit. (e.g. 1 = plane).
    - "gaussian"
        - background calculated from _gaussian fit to image (with rotation)
        - no params required
    - "lorentzian"
        - as above, but a lorentzian lineshape (with rotation)
    - "interpolate"
        - Background defined by the dataset smoothed via a sigma-_gaussian
            filtering, and method-interpolation over masked (polygon) regions.
        - params required in method_params_dict:
            - "interp_method": nearest, linear, cubic.
            - "sigma": sigma passed to _gaussian filter
                (see scipy.ndimage._gaussian_filter) which is utilized on the
                background before interpolating
    - "gaussian_filter"
        - background calculated from image filtered with a _gaussian filter.
        - params required in method_params_dict:
            - "sigma": sigma passed to _gaussian filter (see
                    scipy.ndimage._gaussian_filter)
    - "gaussian_then_poly"
        - runs gaussian then poly subtraction
        - params required in method_params_dict:
            - "order": an int, the 'order' polynomial to fit. (e.g. 1 = plane).

    polygon utilization:
        - if method is not interpolate, the image is masked where the polygons are
          and the background is calculated without these regions
        - if the method is interpolate, these regions are interpolated over
            (and the rest of the image, _gaussian smoothed, is 'background').

    Arguments
    ---------
    image : 2D array-like
        image to get backgrond of
    method : str
        Method to use, available options above
    **method_params_dict : dict
        Key-value pairs passed onto each background backend. Required params
        given above.
    polygons : list, optional
        list of `qdmpy.shared.polygon.Polygon` objects.
        (the default is None, in which case the polygon feature is not used)
    no_bground_if_clip_fails : bool
        You get it.

    Returns
    -------
    im_bground : ndarray
        2D numpy array, representing the 'background' of image.
    mask : ndarray
        Mask (True pixels were not used to calculate background).
    """
    # Discuss masking -> if polygons provided, background is calculated but with
    # polygon regions masked -> background calculated without these.
    # also used for interpolation method.
    method_required_settings: dict[str, list] = {
        "fix_zero": ["zero"],
        "three_point": ["points"],
        "mean": [],
        "poly": ["order"],
        "gaussian": [],
        "lorentzian": [],
        "interpolate": ["interp_method", "sigma"],
        "gaussian_filter": ["sigma"],
        "gaussian_then_poly": ["order"],
    }
    method_fns: dict[str, Callable] = {
        "fix_zero": _zero_background,
        "three_point": _three_point_background,
        "mean": _mean_background,
        "poly": _poly_background,
        "gaussian": _gaussian_background,
        "lorentzian": _lorentzian_background,
        "interpolate": _interpolated_background,
        "gaussian_filter": _filtered_background,
        "gaussian_then_poly": _gaussian_then_poly,
    }
    image = np.ma.array(image, mask=np.zeros(image.shape))
    if len(image.shape) != 2:
        raise ValueError("image is not a 2D array")
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if not isinstance(method_params_dict, dict):
        raise TypeError("method_params_dict must be a dict.")
    if method not in method_required_settings:
        raise ValueError(
            "'method' argument to get_background not in implemented_methods: "
            + f"{method_required_settings.keys()}"
        )
    for setting in method_required_settings[method]:
        if setting not in method_params_dict:
            raise ValueError(
                f"{setting} key missing from method_params_dict for method: {method}"
            )

    if method != "interpolate":
        # can't mask it for interpolate as we need that info!
        image = mask_polygons(image, polygons)

    if method == "gaussian_filter":
        method_params_dict["filter_type"] = "gaussian"

    if method == "interpolate":
        method_params_dict["polygons"] = polygons

    if method == "three_point" and "sample_size" not in method_params_dict:
        method_params_dict["sample_size"] = 0
    return (
        method_fns[method](image, **method_params_dict),
        np.ma.getmaskarray(image).astype(int),
    )


# ============================================================================


def mu_sigma_inside_polygons(
    image: npt.NDArray, polygons: list | None = None
) -> tuple[float, float]:
    """returns (mean, standard_deviation) for image, only _within_ polygon areas."""
    image = mask_polygons(image, polygons, invert_mask=True)
    return np.mean(image), np.std(image)


# ============================================================================


# hands off to other filters
def get_im_filtered(image: npt.NDArray, filter_type: str, **kwargs) -> npt.NDArray:
    """Wrapped over other filters.
    Current filters defined:
        - filter_type = gaussian, `qdmpy.shared.itool._get_im_filtered_gaussian`
    """
    if not isinstance(image, np.ma.core.MaskedArray):
        image = np.ma.masked_array(image)
    filter_fns = {
        "gaussian": _get_im_filtered_gaussian,
    }
    # filter not so great with nans (and mask doesn't work w filters) -> set to nanmean
    if "upper_threshold" in kwargs and kwargs["upper_threshold"]:
        image[image > kwargs["upper_threshold"]] = np.nan
    if "lower_threshold" in kwargs and kwargs["lower_threshold"]:
        image[image < kwargs["lower_threshold"]] = np.nan
    image[np.logical_or(np.isnan(image), image.mask)] = np.nanmean(image)

    return filter_fns[filter_type](image, **kwargs)


# ============================================================================


def _get_im_filtered_gaussian(image: npt.NDArray, sigma: float) -> npt.NDArray:
    """Returns image filtered through scipy.ndimage.gaussian_filter with
    parameter 'sigma'."""
    return scipy.ndimage.gaussian_filter(
        image,
        sigma=sigma,
    )


# ============================================================================


def _zero_background(image: npt.NDArray, zero: float) -> npt.NDArray:
    """Background defined by z level of 'zero'"""
    if not isinstance(zero, (int, float)):
        raise TypeError("'zero' must be an int or a float.")
    mn = zero
    bg = np.empty(image.shape)
    bg[:] = mn
    return bg


# ============================================================================


def _equation_plane(
    params: list | tuple | npt.NDArray, y: npt.NDArray, x: npt.NDArray
) -> npt.NDArray:
    """params: [a, b, c, d] s.t. d = a*y + b*x + c*z
    so z = (1/c) * (d - (ay + bx)) -> return this."""
    return (1 / params[2]) * (params[3] - params[0] * y - params[1] * x)


def _points_to_params(points: list | tuple | npt.NDArray) -> tuple:
    """
    http://pi.math.cornell.edz/~froh/231f08e1a.pdf
    points: iterable of 3 iterables: [x, y, z]
    returns a,b,c,d parameters (see _equation_plane)
    """
    rearranged_points = [[p[1], p[0], p[2]] for p in points]  # change to [y, x, z]
    pts = np.array(rearranged_points)
    vec1_in_plane = pts[1] - pts[0]
    vec2_in_plane = pts[2] - pts[0]
    a_normal = np.cross(vec1_in_plane, vec2_in_plane)

    d = np.dot(pts[2], a_normal)
    return *a_normal, d


def _three_point_background(
    image: npt.NDArray, points: list | tuple | npt.NDArray, sample_size: int
) -> npt.NDArray:
    """points: len 3 iterable of len 2 iterables: [[x1, y1], [x2, y2], [x3, y3]]
    sample_size: integer
    https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points
    https://www.geeksforgeeks.org/program-to-find-equation-of-a-plane-passing-through-3-points/
    """

    if len(points) != 3:
        raise ValueError("points needs to be len 3 of format: [x, y] (int or floats).")
    if not isinstance(sample_size, int) or sample_size < 0:
        raise TypeError("sample_size must be an integer >= 0")
    for p in points:
        if len(p) != 2:
            raise ValueError(
                "points needs to be len 3 of format: [x, y] (int or floats)."
            )
        for c in p:
            if not isinstance(c, (int, float)):
                raise ValueError(
                    "points needs to be len 3 of format: [x, y] (int or floats)."
                )
        if image.mask[p[1], p[0]]:
            dukit_warn(
                "One of the input points was masked (inside a polygon?), "
                + "falling back on polyfit, order 1"
            )
            return _poly_background(image, order=1)

    def _mean_sample(image: npt.NDArray, sample_size: int, yx: tuple) -> float:
        """Mean of samples of 'image' centered on 'yx' out to pixels 'sample_size'
        in both x & y."""

        def _sample_generator(image, sample_size, yx):
            """sample 'image' centered on 'yx' out to pixels 'sample_size'
            in both x & y."""
            for y in range(yx[0] - sample_size, yx[0] + sample_size + 1):
                for x in range(yx[1] - sample_size, yx[1] + sample_size + 1):
                    try:
                        yield image[y, x]
                    except IndexError:
                        continue

        return np.mean(list(_sample_generator(image, sample_size, yx)))

    points = np.array(
        [np.append(p, _mean_sample(image, sample_size, (p[1], p[0]))) for p in points]
    )
    Y, X = np.indices(image.shape)  # noqa: N806
    return _equation_plane(_points_to_params(points), Y, X)


# ============================================================================


def _mean_background(image: npt.NDArray) -> npt.NDArray:
    """Background defined by mean of image."""
    bg = np.empty(image.shape)
    mn = np.nanmean(image)
    bg[:] = mn
    return bg


# ============================================================================


def _residual_poly(
    params: list | tuple | npt.NDArray,
    y: npt.NDArray,
    x: npt.NDArray,
    z: npt.NDArray,
    order: int,
) -> npt.NDArray:
    """
    z = image data, order = highest polynomial order to go to
    y, x: index meshgrids
    """
    # get params to matrix form (as expected by polyval)
    params = np.append(
        params, 0
    )  # add on c[-1, -1] term we don't want (i.e. cross term of next order)
    c = params.reshape((order + 1, order + 1))
    return polyval2d(y, x, c) - z  # note z is flattened, as is polyval


def _poly_background(image: npt.NDArray, order: int) -> npt.NDArray:
    """Background defined by a polynomial fit up to order 'order'."""
    if order == 0:
        return _mean_background(image)

    init_params = np.zeros((order + 1, order + 1))
    init_params[0, 0] = np.nanmean(
        image
    )  # set zeroth term to be mean to get it started
    Y, X = np.indices(image.shape)  # noqa: N806
    good_vals = np.logical_and(~np.isnan(image), ~image.mask)
    y = Y[good_vals]
    x = X[good_vals]

    data = image[good_vals]  # flattened
    best_c = least_squares(
        _residual_poly, init_params.flatten()[:-1], args=(y, x, data, order)
    ).x
    best_c = np.append(best_c, 0)
    c = best_c.reshape((order + 1, order + 1))
    return polyval2d(Y.flatten(), X.flatten(), c).reshape(
        image.shape
    )  # eval over full image


# ============================================================================


def _gaussian(
    p: list | tuple | npt.NDArray, y: npt.NDArray, x: npt.NDArray
) -> npt.NDArray:
    """Simple Gaussian function, height, center_y, center_x, width_y, width_x, rot = p."""
    height, center_y, center_x, width_y, width_x, rot, offset = p
    return offset + height * np.exp(
        -(
            (((y - center_y) * np.cos(rot) + (x - center_x) * np.sin(rot)) / width_y)
            ** 2
            + (((x - center_x) * np.cos(rot) - (y - center_y) * np.sin(rot)) / width_x)
            ** 2
        )
        / 2
    )


def _moments(image: npt.NDArray) -> tuple:
    """Calculate _moments of image (get initial guesses for _gaussian and _lorentzian
    function), rot=0.0 assumed"""
    offset = np.nanmin(image)
    total = np.nansum(image)
    Y, X = np.indices(image.shape)  # noqa: N806

    center_y = np.nansum(Y * image) / total
    center_x = np.nansum(X * image) / total
    if center_y > np.max(Y) or center_y < 0:
        center_y = np.max(Y) / 2
    if center_x > np.max(X) or center_x < 0:
        center_x = np.max(X) / 2

    col = image[int(center_y), :]
    row = image[:, int(center_x)]
    width_x = np.nansum(
        np.sqrt(abs((np.arange(col.size) - center_y) ** 2 * col)) / np.nansum(col)
    )
    width_y = np.nansum(
        np.sqrt(abs((np.arange(row.size) - center_x) ** 2 * row)) / np.nansum(row)
    )
    height = np.nanmax(image)
    return height, center_y, center_x, width_y, width_x, 0.0, offset


def _residual_gaussian(
    p: list | tuple | npt.NDArray, y: npt.NDArray, x: npt.NDArray, data: npt.NDArray
) -> npt.NDArray:
    """Residual of data with a _gaussian model."""
    return _gaussian(p, y, x) - data


def _gaussian_background(image: npt.NDArray) -> npt.NDArray:
    """Background defined by a Gaussian function."""
    params = _moments(image)
    Y, X = np.indices(image.shape)  # noqa: N806
    good_vals = np.logical_and(~np.isnan(image), ~image.mask)
    y = Y[good_vals]
    x = X[good_vals]
    data = image[good_vals]
    p = least_squares(
        _residual_gaussian,
        params,
        method="trf",
        bounds=(0, np.inf),
        args=(y, x, data),
    ).x
    return _gaussian(p, Y, X)


def _lorentzian(
    p: list | tuple | npt.NDArray, y: npt.NDArray, x: npt.NDArray
) -> npt.NDArray:
    height, center_y, center_x, width_y, width_x, rot, offset = p
    xp = (x - center_x) * np.cos(rot) - (y - center_y) * np.sin(rot)
    yp = (x - center_x) * np.sin(rot) + (y - center_y) * np.cos(rot)

    R = (xp / width_x) ** 2 + (yp / width_y) ** 2

    return offset + height * (1 / (1 + R))


def _residual_lorentzian(
    p: list | tuple | npt.NDArray, y: npt.NDArray, x: npt.NDArray, data: npt.NDArray
) -> npt.NDArray:
    return _lorentzian(p, y, x) - data


def _lorentzian_background(image: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    params = _moments(image)
    Y, X = np.indices(image.shape)  # noqa: N806
    good_vals = np.logical_and(~np.isnan(image), ~image.mask)
    y = Y[good_vals]
    x = X[good_vals]
    data = image[good_vals]
    p = least_squares(
        _residual_lorentzian,
        params,
        bounds=(0, np.inf),
        method="trf",
        args=(y, x, data),
    ).x

    return p, _lorentzian(p, Y, X)


# ============================================================================


def _interpolated_background(
    image: npt.NDArray, interp_method: str, polygons: list, sigma: float
) -> npt.NDArray:
    """Background defined by the dataset smoothed via a sigma-_gaussian filtering,
    and method-interpolation over masked (polygon) regions.

    method available: nearest, linear, cubic.
    """
    if not isinstance(polygons, (list, tuple)) or not isinstance(
        polygons[0], dukit.shared.polygon.Polygon
    ):
        raise TypeError("polygons were not None, a list or a list of Polygon objects")

    ylen, xlen = image.shape
    isnt_poly = np.full(image.shape, True)  # all masked to start with

    # coordinate grid for all coordinates
    grid_y, grid_x = np.meshgrid(range(ylen), range(xlen), indexing="ij")

    for p in polygons:
        in_or_out = p.is_inside(grid_y, grid_x)
        # mask all vals that are not background
        is_this_poly = np.ma.masked_greater_equal(
            in_or_out, 0
        ).mask  # >= 0 => inside/on poly
        isnt_poly = np.logical_and(
            isnt_poly, ~is_this_poly
        )  # prev isnt_poly and isnt this poly

    # now we want to send all of the values in indexes that is_bg is True to griddata
    pts = []
    vals = []
    for i in range(ylen):
        for j in range(xlen):
            if isnt_poly[i, j]:
                pts.append([i, j])
                vals.append(image[i, j])

    bg_interp = griddata(pts, vals, (grid_y, grid_x), method=interp_method)

    return get_im_filtered(bg_interp, "gaussian", sigma=sigma)


# ============================================================================


def _filtered_background(image: npt.NDArray, filter_type: str, **kwargs) -> npt.NDArray:
    """Background defined by a filter_type-filtering of the image.
    Passed to `qdmpy.shared.itool.get_background"""
    return get_im_filtered(image, filter_type, **kwargs)


# ============================================================================


def _gaussian_then_poly(image: npt.NDArray, order: int) -> npt.NDArray:
    """Background defines as: fit of a gaussian then a polynomial to image."""
    gauss_back = _gaussian_background(image)
    return _poly_background(image - gauss_back, order) + gauss_back


# ============================================================================
