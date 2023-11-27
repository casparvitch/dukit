# -*- coding: utf-8 -*-
"""
This module holds the general interface tools for fitting data, independent of fit
backend (e.g. scipy/gpufit etc.).

All of these functions are automatically loaded into the namespace when the fit
sub-package is imported. (e.g. import dukit.fit).

Functions
---------
 - `dukit.pl.interface.define_fit_model`
 - `dukit.pl.interface.fit_roi_avg_pl`
 - `dukit.pl.interface.fit_aois_pl`
 - `dukit.pl.interface.fit_all_pixels_pl`
 - `dukit.pl.interface._prep_fit_backends`
 - `dukit.pl.interface.get_pl_fit_result`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.pl.interface.define_fit_model": True,
    "dukit.pl.interface.fit_roi_avg_pl": True,
    "dukit.pl.interface.fit_aois_pl": True,
    "dukit.pl.interface.fit_all_pixels_pl": True,
    "dukit.pl.interface._prep_fit_backends": True,
    "dukit.pl.interface.get_pl_fit_result": True,
}

# ============================================================================

# ============================================================================

# import dukit.pl.model
# import dukit.pl.fastmodel
# import dukit.pl.funcs
# import dukit.pl.io

# ============================================================================

CPUFIT_AVAILABLE: bool = False
try:
    import pycpufit.cpufit as cf
    CPUFIT_AVAILABLE = True
except:
    pass

GPUFIT_AVAILABLE: bool = False
try:
    import pygpufit.gpufit as gf
    GPUFIT_AVAILABLE = True
except:
    pass

# NOTE
# only fastmodel I think. Use init for 8 lorentzians etc.
# fit_backend -> default to fastmodel

# ============================================================================
