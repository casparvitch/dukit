# -*- coding: utf-8 -*-
"""
This module holds tools for fitting raw data via scipy. (scipy backend)

Functions
---------
 - `qdmpy.pl.scipyfit.prep_scipyfit_options`
 - `qdmpy.pl.scipyfit.gen_scipyfit_init_guesses`
 - `qdmpy.pl.scipyfit.fit_roi_avg_pl_scipyfit`
 - `qdmpy.pl.scipyfit.fit_single_pixel_pl_scipyfit`
 - `qdmpy.pl.scipyfit.fit_aois_pl_scipyfit`
 - `qdmpy.pl.scipyfit.limit_cpu`
 - `qdmpy.pl.scipyfit.to_squares_wrapper`
 - `qdmpy.pl.scipyfit.fit_all_pixels_pl_scipyfit`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "qdmpy.pl.scipyfit.prep_scipyfit_options": True,
    "qdmpy.pl.scipyfit.gen_scipyfit_init_guesses": True,
    "qdmpy.pl.scipyfit.fit_roi_avg_pl_scipyfit": True,
    "qdmpy.pl.scipyfit.fit_single_pixel_pl_scipyfit": True,
    "qdmpy.pl.scipyfit.fit_aois_pl_scipyfit": True,
    "qdmpy.pl.scipyfit.limit_cpu": True,
    "qdmpy.pl.scipyfit.to_squares_wrapper": True,
    "qdmpy.pl.scipyfit.fit_all_pixels_pl_scipyfit": True,
}

# ==========================================================================

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from timeit import default_timer as timer
from datetime import timedelta
from joblib import Parallel, delayed
from itertools import product
import logging
from scipy.linalg import svd

# ============================================================================

import dukit.pl.common
import dukit.warn


# ==========================================================================


def prep_scipyfit_options(options, fit_model):
    """
    General options dict -> scipyfit_options
    in format that scipy least_squares expects.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    fit_model : `qdmpy.pl.model.FitModel`
        Fit model object.

    Returns
    -------
    scipy_fit_options : dict
        Dictionary with options that scipy.optimize.least_squares expects, specific to fitting.
    """

    # this is just constructing the initial parameter guesses and bounds in the right format
    _, fit_param_bound_ar = gen_scipyfit_init_guesses(
            options, *dukit.pl.common.gen_init_guesses(options)
    )
    fit_bounds = (fit_param_bound_ar[:, 0], fit_param_bound_ar[:, 1])

    # see scipy.optimize.least_squares
    scipyfit_options = {
        "method": options["scipyfit_method"],
        "verbose": options["scipyfit_verbose_fitting"],
        "gtol": options["scipyfit_fit_gtol"],
        "xtol": options["scipyfit_fit_xtol"],
        "ftol": options["scipyfit_fit_ftol"],
        "loss": options["scipyfit_loss_fn"],
    }

    if options["scipyfit_method"] != "lm":
        scipyfit_options["bounds"] = fit_bounds
        scipyfit_options["verbose"] = options["scipyfit_verbose_fitting"]

    if options["scipyfit_scale_x"]:
        scipyfit_options["x_scale"] = "jac"
    else:
        options["scipyfit_scale_x"] = False

    # define jacobian option for least_squares fitting
    if not fit_model.jacobian_defined() or not options["scipyfit_use_analytic_jac"]:
        scipyfit_options["jac"] = options["scipyfit_fit_jac_acc"]
    else:
        scipyfit_options["jac"] = fit_model.jacobian_scipyfit
    return scipyfit_options


# ==========================================================================


def gen_scipyfit_init_guesses(options, init_guesses, init_bounds):
    """
    Generate arrays of initial fit guesses and bounds in correct form for scipy least_squares.

    init_guesses and init_bounds are dictionaries up to this point, we now convert to np arrays,
    that scipy will recognise. In particular, we specificy that each of the 'num' of each 'fn_type'
    have independent parameters, so must have independent init_guesses and init_bounds when
    plugged into scipy.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    init_guesses : dict
        Dict holding guesses for each parameter, e.g. key -> list of guesses for each independent
        version of that fn_type.
    init_bounds : dict
        Dict holding guesses for each parameter, e.g. key -> list of bounds for each independent
        version of that fn_type.

    Returns
    -------
    fit_param_ar : np array, shape: num_params
        The initial fit parameter guesses.
    fit_param_bound_ar : np array, shape: (num_params, 2)
        Fit parameter bounds.
    """
    param_lst = []
    bound_lst = []

    for fn_type, num in options["fit_functions"].items():
        # extract a guess/bounds for each of the copies of each fn_type (e.g. 8 lorentzians)
        for n in range(num):

            for pos, key in enumerate(qdmpy.pl.funcs.AVAILABLE_FNS[fn_type].param_defn):
                # this check is to handle the edge case of guesses/bounds
                # options being provided as numbers rather than lists of numbers
                try:
                    param_lst.append(init_guesses[key][n])
                except (TypeError, KeyError):
                    param_lst.append(init_guesses[key])
                if len(np.array(init_bounds[key]).shape) == 2:
                    bound_lst.append(init_bounds[key][n])
                else:
                    bound_lst.append(init_bounds[key])

    fit_param_ar = np.array(param_lst)
    fit_param_bound_ar = np.array(bound_lst)
    return fit_param_ar, fit_param_bound_ar


# ==========================================================================


def fit_roi_avg_pl_scipyfit(options, sig, ref, sweep_arr, fit_model):
    """
    Fit the average of the measurement over the region of interest specified, with scipy.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    sig : np array, 3D
        Sig measurement array, unnormalised, shape: [sweep_arr, y, x].
    ref : np array, 3D
        Ref measurement array, unnormalised, shape: [sweep_arr, y, x].
    sweep_arr : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `qdmpy.pl.model.FitModel`
        The fit model object.

    Returns
    -------
    `dukit.pl.common.ROIAvgFitResult` object containing the fit result (see class specifics)
    """
    # fit *all* pl data (i.e. summing over FOV)
    # collapse to just pl_ar (as function of sweep, 1D)
    if not options["used_ref"]:
        roi_norm = sig
    elif options["normalisation"] == "div":
        roi_norm = sig / ref
    elif options["normalisation"] == "sub":
        roi_norm = 1 + (sig - ref) / (sig + ref)
    elif options["normalisation"] == "true_sub":
        roi_norm = (sig - ref) / np.nanmax(sig - ref)

    roi_norm = np.nanmean(roi_norm, axis=(1, 2))

    fit_options = prep_scipyfit_options(options, fit_model)

    init_param_guess, init_bounds = gen_scipyfit_init_guesses(
            options, *dukit.pl.common.gen_init_guesses(options)
    )
    fitting_results = least_squares(
            fit_model.residuals_scipyfit,
            init_param_guess,
            args=(sweep_arr, roi_norm),
            **fit_options,
    )

    best_params = fitting_results.x
    return dukit.pl.common.ROIAvgFitResult(
            "scipyfit",
            fit_options,
            fit_model,
            roi_norm,
            sweep_arr,
            best_params,
            init_param_guess,
            init_bounds,
    )


# ==========================================================================


def fit_single_pixel_pl_scipyfit(
        options, pixel_pl_ar, sweep_arr, fit_model, roi_avg_fit_result
):
    """
    Fit Single pixel and return best_fit_result.x (i.e. the optimal fit parameters)

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    pixel_pl_ar : np array, 1D
        Normalised pl as function of sweep_arr for a single pixel.
    sweep_arr : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `qdmpy.pl.model.FitModel`
        The fit model.
    roi_avg_fit_result : `dukit.pl.common.ROIAvgFitResult`
        `dukit.pl.common.ROIAvgFitResult` object, to pull fit_options from.

    Returns
    -------
    pixel_parameters : np array, 1D
        Best fit parameters, as determined by scipy.
    """

    fit_opts = {}
    for key, val in roi_avg_fit_result.fit_options.items():
        fit_opts[key] = val

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        init_guess_params = roi_avg_fit_result.best_params.copy()
    else:
        # this is just constructing the initial parameter guesses and bounds in the right format
        fit_param_ar, _ = gen_scipyfit_init_guesses(
                options, *dukit.pl.common.gen_init_guesses(options)
        )
        init_guess_params = fit_param_ar.copy()

    fitting_results = least_squares(
            fit_model.residuals_scipyfit,
            init_guess_params,
            args=(sweep_arr, pixel_pl_ar),
            **fit_opts,
    )
    return fitting_results.x


# ==========================================================================


def fit_aois_pl_scipyfit(
        options,
        sig,
        ref,
        pixel_pl_ar,
        sweep_arr,
        fit_model,
        aois,
        roi_avg_fit_result,
):
    """
    Fit AOIs and single pixel and return list of (list of) fit results (optimal fit parameters),
    using scipy.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    sig : np array, 3D
        Sig measurement array, unnormalised, shape: [sweep_arr, y, x].
    ref : np array, 3D
        Ref measurement array, unnormalised, shape: [sweep_arr, y, x].
    single_pixel_pl : np array, 1D
        Normalised measurement array, for chosen single pixel check.
    sweep_arr : np array, 1D
        Affine parameter list (e.g. tau or freq).
    fit_model : `qdmpy.pl.model.FitModel`
        The model we're fitting to.
    aois : list
        List of AOI specifications - each a length-2 iterable that can be used to directly index
        into sig_norm to return that AOI region, e.g. sig_norm[:, AOI[0], AOI[1]].
    roi_avg_fit_result : `dukit.pl.common.ROIAvgFitResult`
        `dukit.pl.common.ROIAvgFitResult` object, to pull `dukit.pl.common.ROIAvgFitResult.fit_options`
        from.

    Returns
    -------
    fit_result_collection : `dukit.pl.common.FitResultCollection`
        Collection of ROI/AOI fit results for this fit backend.
    """
    fit_opts = {}
    for key, val in roi_avg_fit_result.fit_options.items():
        fit_opts[key] = val

    if options["use_ROI_avg_fit_res_for_all_pixels"]:
        guess_params = roi_avg_fit_result.best_params.copy()
    else:
        # this is just constructing the initial parameter guesses and bounds in the right format
        fit_param_ar, _ = gen_scipyfit_init_guesses(
                options, *dukit.pl.common.gen_init_guesses(options)
        )
        guess_params = fit_param_ar.copy()

    single_pixel_fit_params = fit_single_pixel_pl_scipyfit(
            options, pixel_pl_ar, sweep_arr, fit_model, roi_avg_fit_result
    )

    aoi_avg_best_fit_results_lst = []

    for a in aois:
        this_sig = sig[:, a[0], a[1]]
        this_ref = ref[:, a[0], a[1]]

        if not options["used_ref"]:
            this_aoi = this_sig
        elif options["normalisation"] == "div":
            this_aoi = this_sig / this_ref
        elif options["normalisation"] == "sub":
            this_aoi = 1 + (this_sig - this_ref) / (this_sig + this_ref)
        elif options["normalisation"] == "true_sub":
            this_aoi = (this_sig - this_ref) / np.nanmax(this_sig - this_ref)

        fitting_results = least_squares(
                fit_model.residuals_scipyfit,
                guess_params,
                args=(sweep_arr, np.nanmean(this_aoi, axis=(1, 2))),
                **fit_opts,
        )
        aoi_avg_best_fit_results_lst.append(fitting_results.x)

    return dukit.pl.common.FitResultCollection(
            "scipyfit",
            roi_avg_fit_result,
            single_pixel_fit_params,
            aoi_avg_best_fit_results_lst,
    )


# ==========================================================================

def fit_all_pixels_pl_scipyfit(
        sig_norm: npt.NDArray, sweep_arr: npt.NDArray,
        fit_model: dukit.pl.model.FitModel, init_pguess: npt.ArrayLike,
        init_pbounds: tuple[npt.ArrayLike, npt.ArrayLike], n_jobs: int = -2,
        verbose=5, **fit_options
):
    """
    Fits each pixel and returns dictionary of param_name -> param_image.

    Arguments
    ---------
    sig_norm : np array, 3D
        Normalised measurement array, shape: [sweep_arr, y, x].
    sweep_arr : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `dukit.pl.model.FitModel`
        The model we're fitting to.
    init_pguess :
        initial guess on parameters.
    n_jobs : int, default=-2
        Number of jobs to run concurrently, see joblib docs.
        -2 === leaving one cpu free, etc. for neg numbers.
    fit_options
        Other options passed to scipyfit.least_squares

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
        TODO & sigmas!?
    """

    # call into the library (measure time)
    t0 = timer()
    results_lst = Parallel(n_job=n_jobs, verbose=verbose)(
            delayed(_spfitter)(fit_model, sweep_arr, pl_vec,
                               init_pguess, fit_options) for
            pl_vec in iterslice(sig_norm, axis=-1))
    t1 = timer()
    dt = timedelta(seconds=t1 - t0).total_seconds()
    logging.info(f"fit time: {dt:.2f}s")

    results_arr = np.array(results_lst).reshape(
            (*sig_norm.shape[:2], 2 * len(init_pguess) + 1))
    names = fit_model.get_param_defn()
    names.extend([f"{n}_sigma" for n in names])
    names.append("residual_0")
    fit_image_results = {name: array for name, array in
                         zip(names, iterframe(results_arr))}

    return fit_image_results


# =======================================================================================


def _spfitter(fit_model: dukit.pl.model.FitModel,
              sweep_vec: npt.NDArray,
              pl_vec: npt.NDArray,
              p0: npt.NDArray,
              fit_optns: dict) -> npt.NDArray:
    """Single pixel fitter.
    ~ Fit inputs -> (params; param_std_dev; sum(resid)) @ solution.
    """
    try:
        fitres = least_squares(fit_model.residuals_scipyfit, p0,
                               args=(sweep_vec, pl_vec),
                               **fit_optns)
        _, s, vt = svd(fitres.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(fitres.jac.shape) * s[0]
        s = s[s > threshold]
        vt = vt[: s.size]
        pcov = np.dot(vt.T / s ** 2, vt)
        # NOTE using/assuming linear cost fn,
        cost = 2 * 0.5 * np.sum(fit_model(fitres.x, sweep_vec) ** 2)
        resid = fit_model.residuals_scipyfit(pl_vec, fitres.x)
        s_sq = cost / (len(resid) - len(fitres.x))
        pcov *= s_sq
        perr = np.sqrt(np.diag(pcov))  # array of standard deviations
    except ValueError:
        return np.hstack((np.full_like(p0, np.nan),
                          np.full_like(p0, np.nan),
                          np.np.nan))
    return np.hstack((fitres.x, perr, np.sum(
            np.abs(resid, dtype=np.float64), dtype=np.float64
    )))


def iterslice(x: npt.NDArray, axis: int = 0):
    """Iterate through array x in slices along axis 'axis' (defaults to 0).
    E.g. iterslice(shape(y,x,freqs), axis=-1) will give iter. of 1d freq slices."""
    sub = [range(s) for s in x.shape]
    sub[axis] = (slice(None),)
    for p in product(*sub):
        yield x[p]


def iterframe(x_3d: npt.NDArray):
    """iterframe(shape(y,x,freqs)) will give iter. of 2d y,x frames."""
    for frame in range(x_3d.shape[2]):
        yield x_3d[:, :, frame]
