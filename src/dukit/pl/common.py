# -*- coding: utf-8 -*-
"""
This module holds some functions and classes that are shared between different
fitting backends, but are not a part of the user-facing interface.

Functions
---------
 - `dukit.pl.common.shuffle_pixels`
 - `dukit.pl.common.unshuffle_pixels`
 - `dukit.pl.common.unshuffle_fit_results`
 - `dukit.pl.common.pixel_generator`
 - `dukit.pl.common.gen_init_guesses`
 - `dukit.pl.common.bounds_from_range`
 - `dukit.pl.common.get_pixel_fitting_results`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {

    "dukit.pl.common.shuffle_pixels": True,
    "dukit.pl.common.unshuffle_pixels": True,
    "dukit.pl.common.unshuffle_fit_results": True,
    "dukit.pl.common.pixel_generator": True,
    "dukit.pl.common.gen_init_guesses": True,
    "dukit.pl.common.bounds_from_range": True,
    "dukit.pl.common.get_pixel_fitting_results": True,
}

# ============================================================================

from collections.abc import Generator
import numpy as np
import numpy.typing as npt
import copy
from scipy.linalg import svd


# ============================================================================

# import dukit.pl.funcs

# ============================================================================

# ============================================================================


def pixel_generator(our_array: npt.NDArray) -> Generator[tuple]:
    """
    Simple generator to shape data as expected by to_squares_wrapper in scipy
    concurrent method.

    Also allows us to track *where* (i.e. which pixel location) each result corresponds
    to. See also: `dukit.pl.scipyfit.to_squares_wrapper`, and corresponding
    gpufit method.

    Arguments
    ---------
    our_array : np array, 3D
        Shape: [sweep_arr, y, x]

    Returns
    -------
    generator : list
        [y, x, our_array[:, y, x]] generator (yielded)
    """
    _, len_y, len_x = np.shape(our_array)
    for y in range(len_y):
        for x in range(len_x):
            yield y, x, our_array[:, y, x]


# ============================================================================


def gen_init_guesses(options):
    """
    Generate initial guesses (and bounds) in fit parameters from options dictionary.

    Both are returned as dictionaries, you need to use
    'gen_{scipy/gpufit/...}_init_guesses' to convert to the correct (array) format
    for each specific fitting backend.


    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.

    Returns
    -------
    init_guesses : dict
        Dict holding guesses for each parameter, e.g. key -> list of guesses for each
        independent version of that fn_type.
    init_bounds : dict
        Dict holding guesses for each parameter, e.g. key -> list of bounds for each
        independent version of that fn_type.
    """
    init_guesses = {}
    init_bounds = {}

    for fn_type, num in options["fit_functions"].items():
        fit_func = dukit.pl.funcs.AVAILABLE_FNS[fn_type](num)
        for param_key in fit_func.param_defn:
            guess = options[param_key + "_guess"]
            if param_key + "_range" in options:
                bounds = bounds_from_range(options, param_key, guess)
            elif param_key + "_bounds" in options:
                # assumes bounds are passed in with correct formatatting
                bounds = options[param_key + "_bounds"]
            else:
                raise RuntimeError(
                        f"Provide bounds for the {fn_type}.{param_key} param."
                )

            if guess is not None:
                init_guesses[param_key] = guess
                init_bounds[param_key] = np.array(bounds)
            else:
                raise RuntimeError(
                        f"Not sure why your guess for {fn_type}.{param_key} is None?"
                )

    return init_guesses, init_bounds


# ============================================================================


def bounds_from_range(options, param_key, guess):
    """
    Generate parameter bounds (list, len 2) when given a range option.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    param_key : str
        paramater key, e.g. "pos".
    guess : float/int or array
        guess for param, or list of guesses for a given parameter.

    Returns
    -------
    bounds : list
        bounds for each parameter. Dimension depends on dimension of param guess.
    """
    rang = options[param_key + "_range"]
    if isinstance(guess, (list, tuple)) and len(guess) > 1:
        # separate bounds for each fn of this type
        if isinstance(rang, (list, tuple)) and len(rang) > 1:
            bounds = [
                [each_guess - each_range, each_guess + each_range]
                for each_guess, each_range in zip(guess, rang)
            ]
        # separate guess for each fn of this type, all with same range
        else:
            bounds = [
                [
                    each_guess - rang,
                    each_guess + rang,
                ]
                for each_guess in guess
            ]
    else:
        if isinstance(rang, (list, tuple)):
            if len(rang) == 1:
                rang = rang[0]
            else:
                raise RuntimeError("param range len should match guess len")
        # param guess and range are just single vals (easy!)
        else:
            bounds = [
                guess - rang,
                guess + rang,
            ]
    return bounds


# ============================================================================

def get_pixel_fitting_results(fit_model, results_lst, pixel_data, sweep_arr):
    """
    Take the fit result data from scipyfit/gpufit and back it down to a dictionary of arrays.

    Each array is 2D, representing the values for each parameter (specified by the dict key).


    Arguments
    ---------
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fitting to.
    fit_results : list of [(y, x), result, jac] objects
        (see `qdmpy.pl.scipyfit.to_squares_wrapper`, or corresponding gpufit method)
        A list of each pixel's parameter array, as well as position in image denoted by (y, x).
    pixel_data : np array, 3D
        Normalised measurement array, shape: [sweep_arr, y, x]. i.e. sig_norm.
        May or may not already be shuffled (i.e. matches fit_results).
    sweep_arr : np array, 1D
        Affine parameter list (e.g. tau or freq).

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    sigmas : dict
        Dictionary, key: param_keys, val: image (2D) of param uncertainties across FOV.
    """

    roi_shape = np.shape(pixel_data)[1:]

    # initialise dictionary with key: val = param_name: param_units
    fit_image_results = fit_model.get_param_odict()
    sigmas = copy.copy(fit_image_results)

    # override with correct size empty arrays using np.zeros
    for key in fit_image_results.keys():
        fit_image_results[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan
        sigmas[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    fit_image_results["residual_0"] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    # Fill the arrays element-wise from the results function, which returns a
    # 1D array of flattened best-fit parameters.
    for (y, x), result, jac in fit_results:
        filled_params = {}  # keep track of index, i.e. pos_0, for this pixel

        if jac is None:  # can't fit this pixel
            fit_image_results["residual_0"][y, x] = np.nan
            perr = np.empty(np.shape(result))
            perr[:] = np.nan
        else:
            # NOTE we decide not to call each backend separately here
            resid = fit_model.residuals_scipyfit(result, sweep_arr, pixel_data[:, y, x])
            fit_image_results["residual_0"][y, x] = np.sum(
                    np.abs(resid, dtype=np.float64), dtype=np.float64
            )
            # uncertainty (covariance matrix), copied from scipy.optimize.curve_fit (not abs. sigma)
            _, s, vt = svd(jac, full_matrices=False)
            threshold = np.finfo(float).eps * max(jac.shape) * s[0]
            s = s[s > threshold]
            vt = vt[: s.size]
            pcov = np.dot(vt.T / s ** 2, vt)
            # NOTE using/assuming linear cost fn,
            cost = 2 * 0.5 * np.sum(fit_model(result, sweep_arr) ** 2)
            s_sq = cost / (len(resid) - len(result))
            pcov *= s_sq
            perr = np.sqrt(np.diag(pcov))  # array of standard deviations

        for param_num, param_name in enumerate(fit_model.get_param_defn()):
            # keep track of what index we're up to, i.e. pos_1
            if param_name not in filled_params.keys():
                key = param_name + "_0"
                filled_params[param_name] = 1
            else:
                key = param_name + "_" + str(filled_params[param_name])
                filled_params[param_name] += 1

            fit_image_results[key][y, x] = result[param_num]
            sigmas[key][y, x] = perr[param_num]

    return fit_image_results, sigmas
