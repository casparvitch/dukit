# -*- coding: utf-8 -*-
"""
TODO full docs.

Import/dep graph
----------------

![](../../dukit.png)
"""

from ._version import __version__

from dukit.systems import (
    System,
    MelbSystem,
    LVControl,
    PyControl,
    Zyla,
    CryoWidefield,
    LegacyCryoWidefield,
    Argus,
    LegacyArgus,
    PyCryoWidefield,
)
from dukit.itool import (
    mpl_set_run_config,
    crop_sweep,
    crop_roi,
    smooth_image_stack,
    rebin_image_stack,
    sum_spatially,
    get_im_filtered,
    get_background,
    get_colormap_range,
    plot_image,
    plot_image_on_ax,
    mask_polygons,
    get_im_filtered,
    get_background,
    mu_sigma_inside_polygons,
    get_aois,
)

from dukit.widget import (
    LineSelector,
    PolygonSelector,
    LinecutSelectionWidget,
    BulkLinecutWidget,
)
from dukit.polygon import (
    Polygon,
    PolygonSelectionWidget,
    polygon_selector,
    load_polygon_nodes,
)

from dukit.geom import (
    get_unvs,
    get_unv_frames,
    NV_AXES_111,
    NV_AXES_100_100,
    NV_AXES_100_110,
)

from dukit.fourier import (
    define_k_vectors,
    hanning_filter_kspace,
    define_current_transform,
    define_magnetization_transformation,
    pad_image,
    unpad_image,
    MAG_UNIT_CONV,
    MU_0,
)

from dukit.driftcorrect import drift_correct_test, drift_correct_measurement

from dukit.magsim import SandboxMagSim, ComparisonMagSim

# from dukit.share import RoiAvgFit, AoiAvgFit

import dukit.plot

# === PL fitting etc.

from dukit.pl import (
    FitModel,
    ConstStretchedExp,
    ConstDampedRabi,
    LinearLorentzians,
    ConstLorentzians,
    fit_all_pixels,
    fit_roi,
    fit_aois,
    load_fit_results,
)

# === field stuff
from dukit.field import (
    Defect,
    SpinOne,
    NVEnsemble,
    VBEnsemble,
    SpinPair,
    CSpinPair,
)
