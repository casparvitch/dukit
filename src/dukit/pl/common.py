# -*- coding: utf-8 -*-
"""
This module holds some functions and classes that are shared between different
fitting backends, but are not a part of the user-facing interface.

Functions
---------
 - `dukit.pl.common.gen_init_guesses`
 - `dukit.pl.common.bounds_from_range`
 - `dukit.pl.common.calc_sigmas`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.pl.common.gen_init_guesses": True,
    "dukit.pl.common.bounds_from_range": True,
    "dukit.pl.common.calc_sigmas": True,
}

# ============================================================================

import numpy as np
import numpy.typing as npt
from scipy.linalg import svd
from itertools import product

# ============================================================================

# ============================================================================


def gen_init_guesses(
    fit_model: "dukit.pl.model.FitModel", guesses_dict: dict, bounds_dict: dict
):
    """
    Generate initial guesses (and bounds) in fit parameters from options dictionary.

    Both are returned as dictionaries, you need to use
    'gen_{scipy/gpufit/...}_init_guesses' to convert to the correct (array) format
    for each specific fitting backend.


    Arguments
    ---------
    fit_model : dukit.pl.model.FitModel
        Fit model you've defined already.
    guesses_dict : dict
        Format: key -> list of guesses for each independent version of that fn_type.
        e.g. 'pos': [.., ..] for each pos fn_type.
    bounds_dict : dict
        Format: key -> bounds for that param type (or use _range).
        e.g. 'pos_bounds': [5., 25.]
        or 'pos_range': 5.0
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

    for param_key in fit_model.get_param_defn():
        guess = guesses_dict[param_key]

        if param_key + "_range" in bounds_dict:
            bounds = bounds_from_range(bounds_dict[param_key + "_range"], guess)
        elif param_key + "_bounds" in bounds_dict:
            bounds = bounds_dict[param_key + "_bounds"]
        else:
            raise RuntimeError(f"Provide bounds for {param_key}!")

        if guess is not None:
            init_guesses[param_key] = guess
            init_bounds[param_key] = np.asarray(bounds)
        else:
            raise RuntimeError(f"Not sure why your guess for {param_key} is None?")

    return init_guesses, init_bounds


# ============================================================================


def bounds_from_range(
    rang: float | npt.ArrayLike, guess: float | npt.ArrayLike
) -> tuple:
    """
    Generate parameter bounds when given a range option.

    Arguments
    ---------
    rang : float or npt.ArrayLike
        Range for each parameter with name param_key e.g. 'pos_0, pos_1', OR
        a single value, so each parameter has same range.
    guess : float/int or array
        Guess for param, or list of guesses for a given parameter (as for rang)

    Returns
    -------
    bounds : tuple
        bounds for each parameter. Dimension depends on dimension of param guess.
    """
    if isinstance(guess, (list, tuple)) and len(guess) > 1:
        # separate bounds for each fn of this type
        if isinstance(rang, (list, tuple)) and len(rang) > 1:
            bounds = (
                (each_guess - each_range, each_guess + each_range)
                for each_guess, each_range in zip(guess, rang)
            )
        # separate guess for each fn of this type, all with same range
        else:
            bounds = (
                (
                    each_guess - rang,
                    each_guess + rang,
                )
                for each_guess in guess
            )
    else:
        if isinstance(rang, (list, tuple)):
            if len(rang) == 1:
                rang = rang[0]
            else:
                raise RuntimeError("param range len should match guess len")
        # param guess and range are just single vals (easy!)
        bounds = (
            guess - rang,
            guess + rang,
        )
    return list(bounds)


# ============================================================================


def calc_sigmas(
    fit_model: "dukit.pl.model.FitModel",
    sweep_arr: npt.NDArray,
    pl_vec: npt.NDArray,
    best_params: npt.NDArray,
) -> npt.NDArray:
    """Calculate fit errors (std. dev.) from jacobian.

    Arguments
    ---------
    fit_model : dukit.pl.model.FitModel
        Fit model you've defined already.
    sweep_arr : npt.NDArray, 1D
        Array of sweep values.
    pl_vec : npt.NDArray, 1D
        Array of measured PL values.
    best_params : npt.NDArray, 1D
        Array of best-fit parameters.

    Returns
    -------
    sigmas : npt.NDArray, 1D
        Array of standard deviations for each parameter.
    """
    jac = fit_model.jacobian_scipyfit(best_params, sweep_arr, pl_vec)
    _, s, vt = svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    vt = vt[: s.size]
    pcov = np.dot(vt.T / s**2, vt)
    # NOTE using/assuming linear cost fn,
    cost = 2 * 0.5 * np.sum(fit_model(best_params, sweep_arr) ** 2)
    resid = fit_model.residuals_scipyfit(best_params, sweep_arr, pl_vec)
    s_sq = cost / (len(resid) - len(best_params))
    pcov *= s_sq
    return np.sqrt(np.diag(pcov))  # array of standard deviations

# ============================================================================

def _iterslice(x: npt.NDArray, axis: int = 0):
    """Iterate through array x in slices along axis 'axis' (defaults to 0).
    E.g. iterslice(shape(y,x,freqs), axis=-1) will give iter. of 1d freq slices."""
    sub = [range(s) for s in x.shape]
    sub[axis] = (slice(None),)
    for p in product(*sub):
        yield x[p]

# ============================================================================

def _iterframe(x_3d: npt.NDArray):
    """iterframe(shape(y,x,freqs)) will give iter. of 2d y,x frames."""
    for frame in range(x_3d.shape[2]):
        yield x_3d[:, :, frame]
