# -*- coding: utf-8 -*-
"""
Module for fitting PL data.

Cpufit/Gpufit are imported if available.

Classes
-------
- `dukit.pl.model.FitModel`
- `dukit.pl.model.ConstStretchedExp`
- `dukit.pl.model.ConstDampedRabi`
- `dukit.pl.model.LinearLorentzians`
- `dukit.pl.model.ConstLorentzians`

Functions
---------
- `dukit.pl.interface.fit_all_pixels`
- `dukit.pl.interface.fit_roi`
- `dukit.pl.interface.fit_aois`
- `dukit.pl.interface.load_fit_results`
"""

from dukit.pl.model import (
    FitModel,
    ConstStretchedExp,
    ConstDampedRabi,
    LinearLorentzians,
    ConstLorentzians,
)

from dukit.pl.interface import fit_all_pixels, fit_roi, fit_aois, load_fit_results