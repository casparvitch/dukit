# -*- coding: utf-8 -*-
"""
Data structures/types shared between other modules.

Classes
-------
 - `dukit.share.AoiAvgFit`
 - `dukit.share.RoiAvgFit`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.share.AoiAvgFit": True,
    "dukit.share.RoiAvgFit": True,
}

# ============================================================================

import numpy.typing as npt

# ============================================================================

import dukit.json2dict


# ============================================================================
class AoiAvgFit:
    """Holds result from an AOI fit. Only a method to save.

    Attributes
    ----------
    aoi_num : int
        Number associated with this AOI
    sweep_arr : ndarray, 1D
        Freqs/taus
    aoi_avg_pl : ndarray, 1D
        PL (trace/spectrum) averaged over this aoi
    fit_backend : str
        Name of the fit backend (e.g. scipy, gpufit, cpufit, etc.) used.
    best_params : ndarray, 1D
        Best (optimal) fit/model parameters
    init_pguess: tuple of floats
        Guesses for fit parameters
    init_pbounds : 2-tuple, of tuples
        Bounds for fit, first tuple are the lower bounds, second the upper bounds.
    fit_options : dict
        Other options passed to fit(s).

    Methods
    -------
    save_json(filepath)
        Save into given filepath, ending in '.json'.
    """

    def __init__(
            self,
            aoi_num: int,
            sweep_arr: npt.NDArray,
            aoi_avg_pl: npt.NDArray,
            fit_backend: str,
            best_params: npt.NDArray,
            init_pguess: tuple[float],
            init_pbounds: tuple[tuple, tuple],
            aoi_coords: tuple[int, int, int, int],
            fit_options: dict,
    ):
        """
        Arguments
        ---------
        aoi_num : int
            Number associated with this AOI
        sweep_arr : ndarray, 1D
            Freqs/taus
        aoi_avg_pl : ndarray, 1D
            PL (trace/spectrum) averaged over this AOI
        fit_backend : str
            Name of the fit backend (e.g. scipy, gpufit, cpufit, etc.) used.
        best_params : ndarray, 1D
            Best (optimal) fit/model parameters
        init_pguess: tuple of floats
            Guesses for fit parameters
        init_pbounds : 2-tuple, of tuples
            Bounds for fit, first tuple are the lower bounds, second the upper bounds.
        aoi_coords : 4-tuple of ints
            start_x, start_y, end_x, end_y
        fit_options : dict
            Other options passed to fit(s).
        """
        self.aoi_num = aoi_num
        self.sweep_arr = sweep_arr
        self.aoi_avg_pl = aoi_avg_pl
        self.fit_backend = fit_backend
        self.best_params = best_params
        self.init_pguess = init_pguess
        self.init_pbounds = init_pbounds
        self.fit_options = fit_options
        self.aoi_coords = aoi_coords

    def save_json(self, filepath):
        """
        Save all attributes as a json file in filepath (ending in '.json').
        """
        output_dict = {
            "aoi_num": self.aoi_num,
            "sweep_arr": self.sweep_arr,
            "aoi_avg_pl": self.aoi_avg_pl,
            "fit_backend": self.fit_backend,
            "best_params": self.best_params,
            "init_pguess": self.init_pguess,
            "init_pbounds": self.init_pbounds,
            "fit_options": self.fit_options,
            "aoi_coords": self.aoi_coords,
        }
        dukit.json2dict.dict_to_json(output_dict, filepath)


# ============================================================================


class RoiAvgFit:
    """Holds result from an ROI fit. Only a method to save.

    Attributes
    ----------
    fit_backend : str
        Name of the fit backend (e.g. scipy, gpufit, cpufit, etc.) used.
    sweep_arr : ndarray, 1D
        Freqs/taus
    roi_avg_pl : ndarray, 1D
        PL (trace/spectrum) averaged over ROI
    best_params : ndarray, 1D
        Best (optimal) fit/model parameters
    init_pguess: tuple of floats
        Guesses for fit parameters
    init_pbounds : 2-tuple, of tuples
        Bounds for fit, first tuple are the lower bounds, second the upper bounds.
    roi_coords : 4-tuple of ints
        start_x, start_y, end_x, end_y
    fit_options : dict
        Other options passed to fit(s).

    Methods
    -------
    save_json(filepath)
        Save into given filepath, ending in '.json'.
    """

    def __init__(
            self,
            fit_backend,
            sweep_ar,
            roi_avg_pl,
            best_params,
            init_pguess,
            init_pbounds,
            roi_coords,
            fit_options,
    ):
        """
        Arguments
        ---------
        fit_backend : str
            Name of the fit backend (e.g. scipy, gpufit, cpufit, etc.) used.
        sweep_arr : ndarray, 1D
            Freqs/taus
        roi_avg_pl : ndarray, 1D
            PL (trace/spectrum) averaged over ROI
        best_params : ndarray, 1D
            Best (optimal) fit/model parameters
        init_pguess: tuple of floats
            Guesses for fit parameters
        init_pbounds : 2-tuple, of tuples
            Bounds for fit, first tuple are the lower bounds, second the upper bounds.
        roi_coords : 4-tuple of ints
            start_x, start_y, end_x, end_y
        fit_options : dict
            Other options passed to fit(s).
        """
        self.fit_backend = fit_backend
        self.sweep_ar = sweep_ar
        self.roi_avg_pl = roi_avg_pl
        self.best_params = best_params
        self.init_pguess = init_pguess
        self.init_pbounds = init_pbounds
        self.fit_options = fit_options
        self.roi_coords = roi_coords

    def save_json(self, filepath):
        """
        Save all attributes as a json file in filepath (ending in '.json').
        """
        output_dict = {
            "fit_backend": self.fit_backend,
            "sweep_ar": self.sweep_ar,
            "roi_avg_pl": self.roi_avg_pl,
            "best_params": self.best_params,
            "init_pguess": self.init_pguess,
            "init_pbounds": self.init_pbounds,
            "fit_options": self.fit_options,
            "roi_coords": self.roi_coords,
        }
        dukit.json2dict.dict_to_json(output_dict, filepath)
