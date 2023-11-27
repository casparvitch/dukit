# -*- coding: utf-8 -*-
"""
This module holds functions for plotting initial processing images and fit results.

Functions
---------
 - `dukit.plot.pl.roi_pl_image`
 - `dukit.plot.pl.aoi_pl_image`
 - `dukit.plot.pl.roi_avg_fits`
 - `dukit.plot.pl.aoi_spectra`
 - `dukit.plot.pl.aoi_spectra_fit`
 - `dukit.plot.pl.pl_param_image`
 - `dukit.plot.pl.pl_param_images`
 - `dukit.plot.pl._add_patch_rect`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.plot.pl.roi_pl_image": True,
    "dukit.plot.pl.aoi_pl_image": True,
    "dukit.plot.pl.roi_avg_fits": True,
    "dukit.plot.pl.aoi_spectra": True,
    "dukit.plot.pl.aoi_spectra_fit": True,
    "dukit.plot.pl.pl_param_image": True,
    "dukit.plot.pl.pl_param_images": True,
    "dukit.plot.pl._add_patch_rect": True,
}

# ============================================================================

import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patches
import numpy as np
import numpy.typing as npt

# ============================================================================

import dukit.pl
import dukit.shared.plot
import dukit.shared.json2dict
import dukit.shared.misc
# import dukit.shared.plot
# import dukit.shared.json2dict
# import dukit.shared.misc

# ===========================================================================

AOI_COLORS = [
    "blue",
    "tab:brown",
    "purple",
    "darkslategrey",
    "magenta",
    "olive",
    "cyan",
]

FIT_BACKEND_COLORS = {
    "scipyfit": {
        "ROIfit_linecolor": "mediumblue",
        "residual_linecolor": "black",
        "AOI_ROI_fit_linecolor": "indigo",
        "AOI_best_fit_linecolor": "crimson",
    },
    "gpufit": {
        "ROIfit_linecolor": "cornflowerblue",
        "residual_linecolor": "dimgrey",
        "AOI_ROI_fit_linecolor": "mediumpurple",
        "AOI_best_fit_linecolor": "coral",
    },
    "cpufit": {
        "ROIfit_linecolor": "lightskyblue",
        "residual_linecolor": "lightgrey",
        "AOI_ROI_fit_linecolor": "plum",
        "AOI_best_fit_linecolor": "goldenrod",
    },
}


# ===========================================================================


def roi_pl_image(
        pl_image: npt.NDArray,
        roi_coords: tuple[int, int, int, int],
        opath: str = "",
        **kwargs,
):
    """
    Plots full pl image with ROI region annotated.

    Arguments
    ---------
    pl_image : np array, 2D
        UNCCROPPED pl image, but binned
    roi_coords : tuple[int, int, int, int]
        ROI coordinates
    opath : str
        If supplied, saves figure here.
    **kwargs
        Plotting options passed to `dukit.shared.plot.plot_image_on_ax`
    Returns
    -------
    fig : plt.Figure

    ax : plt.Axes
    """

    fig, ax = plt.subplots()
    if "c_range_type" in kwargs and "c_range_vals" in kwargs:
        c_range = dukit.shared.plot.get_colormap_range(
                kwargs["c_range_type"], kwargs["c_range_vals"], pl_image
        )
    else:
        c_range = dukit.shared.plot.get_colormap_range("min_max", (), pl_image)

    if "c_map" in kwargs:
        c_map = kwargs["c_map"]
        del kwargs["c_map"]
    else:
        c_map = "grey"

    fig, ax = dukit.shared.plot.plot_image_on_ax(
            fig,
            ax,
            pl_image,
            title="PL - Full & Rebinned",
            c_range=c_range,
            c_label="Counts",
            c_map=c_map,
            **kwargs,
    )

    _add_patch_rect(ax, roi_coords, label="ROI", edgecolor="r")
    if opath:
        fig.savefig(opath)


# ============================================================================


def aoi_pl_image(
        pl_image: npt.NDArray,
        *aoi_coords: tuple[int, int, int, int],
        opath: str = "",
        **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots pl image cut down to ROI, with annotated AOI regions.

    Arguments
    ---------
    pl_image : np array, 2D
        pl image AFTER cropping.
    *aoi_coords : tuple[int, int, int, int]
        AOI coordinates
    opath : str
        If supplied, saves figure here.
    **kwargs
        Plotting options passed to `dukit.shared.plot.plot_image_on_ax`

    Returns
    -------
    fig : plt.Figure

    ax : plt.Axes

    """
    fig, ax = plt.subplots()
    if "c_range_type" in kwargs and "c_range_vals" in kwargs:
        c_range = dukit.shared.plot.get_colormap_range(
                kwargs["c_range_type"], kwargs["c_range_vals"], pl_image
        )
    else:
        c_range = dukit.shared.plot.get_colormap_range("min_max", (), pl_image)

    if "c_map" in kwargs:
        c_map = kwargs["c_map"]
        del kwargs["c_map"]
    else:
        c_map = "grey"

    fig, ax = dukit.shared.plot.plot_image_on_ax(
            fig,
            ax,
            pl_image,
            title="PL - ROI & Rebinned",
            c_range=c_range,
            c_label="Counts",
            c_map=c_map,
            **kwargs,
    )

    # add single pixel aoi first
    shp = pl_image.shape
    _add_patch_rect(ax, (shp[0] // 2, shp[1] // 2, shp[0] // 2 + 1, shp[1] // 2 + 1),
                    label="AOI 0", edgecolor=AOI_COLORS[0])
    for i, aoi in enumerate(aoi_coords):
        _add_patch_rect(ax, aoi, label="AOI " + str(i + 1), edgecolor=AOI_COLORS[i + 1])

    if opath:
        fig.savefig(opath)

    return fig, ax


# ============================================================================


def roi_avg_fits(roi_results: dict[str, dukit.pl.common.RoiAvgFit],
                 opath: str = ""):
    """
    Plots fit of spectrum averaged across ROI, as well as corresponding residual values.

    Arguments
    ---------
    roi_results : dict[str, dukit.pl.common.RoiAvgFit
        dict["fit_backend"] => RoiAvgFit
    opath : str
        If given, save figure here.
    Returns
    -------
    fig : matplotlib Figure object
    """
    figsize = mpl.rcParams["figure.figsize"].copy()
    figsize[0] *= 2
    figsize[1] *= 1.5

    fig = plt.figure(figsize=figsize)
    # xstart, ystart, xend, yend
    # [units are fraction of the image frame, from bottom left corner]
    spectrum_frame = fig.add_axes((0.1, 0.3, 0.8, 0.6))

    lspec_names = []
    lspec_lines = []

    spectrum_frame.plot(
            roi_results[0].sweep_list,
            roi_results[0].pl_roi,
            label="raw data",
            ls=" ",
            marker="o",
            mfc="w",
            mec="firebrick",
    )
    lspec_names.append("raw data")
    lspec_lines.append(Line2D([0], [0], ls=" ", marker="o", mfc="w", mec="firebrick"))

    arb_result = next(iter(roi_results.values()))
    high_res_sweep_list = np.linspace(
            np.min(arb_result.sweep_list),
            np.max(arb_result.sweep_list),
            10000,
    )
    high_res_init_fit = arb_result.fit_model(
            arb_result.init_param_guess, high_res_sweep_list
    )
    spectrum_frame.plot(
            high_res_sweep_list,
            high_res_init_fit,
            linestyle=(0, (1, 1)),
            label="init guess",
            c="darkgreen",
    )
    lspec_names.append("init guess")
    lspec_lines.append(Line2D([0], [0], linestyle=(0, (1, 1)), c="darkgreen"))

    plt.setp(spectrum_frame.get_xticklabels(), visible=False)

    spectrum_frame.grid()
    spectrum_frame.set_ylabel("pl (a.u.)")

    # residual plot
    residual_frame = fig.add_axes((0.1, 0.1, 0.8, 0.2), sharex=spectrum_frame)

    lresid_names = []
    lresid_lines = []

    residual_frame.grid()

    residual_frame.set_xlabel("Sweep parameter")
    residual_frame.set_ylabel("Fit - data (a.u.)")

    for _, res in roi_results.items():
        # ODMR spectrum_frame
        high_res_best_fit = res.fit_model(res.best_params, high_res_sweep_list)

        spectrum_frame.plot(
                high_res_sweep_list,
                high_res_best_fit,
                linestyle="--",
                label=f"{res.fit_backend} best fit",
                c=FIT_BACKEND_COLORS[res.fit_backend]["ROIfit_linecolor"],
        )
        lspec_names.append(f"{res.fit_backend} best fit")
        lspec_lines.append(
                Line2D(
                        [0],
                        [0],
                        linestyle="--",
                        c=FIT_BACKEND_COLORS[res.fit_backend]["ROIfit_linecolor"],
                )
        )

        residual_xdata = res.sweep_list
        residual_ydata = res.fit_model(res.best_params, res.sweep_list) - res.pl_roi

        residual_frame.plot(
                residual_xdata,
                residual_ydata,
                label=f"{res.fit_backend} residual",
                ls="dashed",
                c=FIT_BACKEND_COLORS[res.fit_backend]["residual_linecolor"],
                marker="o",
                mfc="w",
                mec=FIT_BACKEND_COLORS[res.fit_backend]["residual_linecolor"],
        )
        lresid_names.append(f"{res.fit_backend} residual")
        lresid_lines.append(
                Line2D(
                        [0],
                        [0],
                        ls="dashed",
                        c=FIT_BACKEND_COLORS[res.fit_backend]["residual_linecolor"],
                        marker="o",
                        mfc="w",
                        mec=FIT_BACKEND_COLORS[res.fit_backend]["residual_linecolor"],
                )
        )

        # FIXME I think we should do this in the fitting call, not here.
        # res.savejson(f"ROI_avg_fit_{res.fit_backend}.json", options["data_dir"])

    # https://jdhao.github.io/2018/01/23/matplotlib-legend-outside-of-axes/
    # https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/linestyles.html
    legend_names = lspec_names.copy()
    legend_names.extend(lresid_names)

    legend_lines = lspec_lines.copy()
    legend_lines.extend(lresid_lines)

    spectrum_frame.legend(
            legend_lines,
            legend_names,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            fontsize="medium",
            ncol=len(legend_names),
            borderaxespad=0,
            frameon=False,
    )
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # spectrum_frame.legend()

    if opath:
        fig.savefig(opath)

    return fig


# ============================================================================


def aoi_spectra(
        sig: npt.NDArray,
        ref: npt.NDArray,
        sweep_arr: npt.NDArray,
        *aoi_coords: tuple[int, int, int, int],
        specpath: str = "",
        opath: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots spectra from each AOI, as well as subtraction and division norms.

    Arguments
    ---------
    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    ref : np array, 3D
        Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    sweep_arr : ndarray
        List of sweep parameter values (with removed unwanted sweeps at start/end)
    specpath : str
        Path (preferably to json) to save spectra in
    opath : str
        Path to save figure in.

    Returns
    -------
    fig : matplotlib Figure object
    """

    aois = _get_aois(sig.shape, *aoi_coords)  # type: ignore

    # pre-process data to plot
    sig_avgs = []
    ref_avgs = []
    for i, aoi in enumerate(aois):
        sig_avg = np.nanmean(sig[:, aoi[0], aoi[1]], axis=(1, 2))
        ref_avg = np.nanmean(ref[:, aoi[0], aoi[1]], axis=(1, 2))
        sig_avgs.append(sig_avg)
        ref_avgs.append(ref_avg)

    figsize = mpl.rcParams["figure.figsize"].copy()
    num_wide = 3 if len(aois) < 3 else len(aois)
    figsize[0] *= 0.6*num_wide
    figsize[1] *= 1.75
    fig, axs = plt.subplots(2, num_wide, figsize=figsize, sharex=True, sharey=False)

    for i, aoi in enumerate(aois):
        # plot sig
        axs[0, i].plot(
                sweep_arr,
                sig_avgs[i],
                label="sig",
                c="blue",
                ls="dashed",
                marker="o",
                mfc="cornflowerblue",
                mec="mediumblue",
        )
        # plot ref
        axs[0, i].plot(
                sweep_arr,
                ref_avgs[i],
                label="ref",
                c="green",
                ls="dashed",
                marker="o",
                mfc="limegreen",
                mec="darkgreen",
        )

        axs[0, i].legend()
        axs[0, i].grid(True)
        axs[0, i].set_title(
                "AOI " + str(i),
                fontdict={"color": AOI_COLORS[i]},
        )
        axs[0, i].set_ylabel("pl (a.u.)")

    linestyles = [
        "--",
        "-.",
        (0, (1, 1)),
        (0, (5, 10)),
        (0, (5, 5)),
        (0, (5, 1)),
        (0, (3, 10, 1, 10)),
        (0, (3, 5, 1, 5)),
        (0, (3, 1, 1, 1)),
        (0, (3, 5, 1, 5, 1, 5)),
        (0, (3, 10, 1, 10, 1, 10)),
        (0, (3, 1, 1, 1, 1, 1)),
    ]

    for i in range(len(aois)):
        # plot subtraction norm
        axs[1, 0].plot(
                sweep_arr,
                1 + (sig_avgs[i] - ref_avgs[i]) / (sig_avgs[i] + ref_avgs[i]),
                label="AOI " + str(i + 1),
                c=AOI_COLORS[i],
                ls=linestyles[i],
                marker="o",
                mfc="w",
                mec=AOI_COLORS[i],
        )
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].set_title(
                "Sub. Norm. (Michelson contrast, 1 + (sig - ref / sig +"
                " ref) )"
        )
        axs[1, 0].set_xlabel("Sweep parameter")
        axs[1, 0].set_ylabel("pl (a.u.)")

        # plot division norm
        axs[1, 1].plot(
                sweep_arr,
                sig_avgs[i] / ref_avgs[i],
                label="AOI " + str(i),
                c=AOI_COLORS[i],
                ls=linestyles[i],
                marker="o",
                mfc="w",
                mec=AOI_COLORS[i],
        )
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].set_title("Div. Norm. (Weber contrast, sig / ref)")
        axs[1, 1].set_xlabel("Sweep parameter")
        axs[1, 1].set_ylabel("pl (a.u.)")

        # plot true-sub norm
        axs[1, 2].plot(
                sweep_arr,
                (sig_avgs[i] - ref_avgs[i]) - np.nanmax(sig_avgs[i] - ref_avgs[i],
                                                        axis=0),
                label="AOI " + str(i),
                c=AOI_COLORS[i],
                ls=linestyles[i],
                marker="o",
                mfc="w",
                mec=AOI_COLORS[i],
        )
        axs[1, 2].legend()
        axs[1, 2].grid(True)
        axs[1, 2].set_title("True-sub Norm. (sig - ref / mx)")
        axs[1, 2].set_xlabel("Sweep parameter")
        axs[1, 2].set_ylabel("pl (a.u.)")

    # delete axes that we didn't use
    for i in range(len(aois)):
        if i < 3:  # we used these (3 normalisation types)
            continue
        else:  # we didn't use these
            fig.delaxes(axs[1, i])

    if len(aois) == 2:
        fig.delaxes(axs[0, 2])
    if len(aois) == 1:
        fig.delaxes(axs[0, 1])
        fig.delaxes(axs[0, 2])

    output_dict = {"sweep_arr": sweep_arr}
    for i in range(len(aois)):
        output_dict["AOI_sig_avg" + "_" + str(i)] = sig_avgs[i]
        output_dict["AOI_ref_avg" + "_" + str(i)] = ref_avgs[i]

    if specpath:
        dukit.shared.json2dict.dict_to_json(
                output_dict,
                specpath,
        )

    if opath:
        fig.savefig(opath)

    return fig, axs


# ============================================================================


# TODO add type for fit_model
def aoi_spectra_fit(
        sig: npt.NDArray,
        ref: npt.NDArray,
        sweep_list: npt.NDArray,
        fit_result_lst,  # list[dukit.pl.common.FitResult],
        fit_model,
        opath: str,
        norm: str,
        *aoi_coords: tuple[int, int, int, int],
):
    """
    Plots sig and ref spectra, sub and div normalisation and fit for the ROI average,
    a single pixel, and each of the AOIs. Stacked on top of each other for comparison.
    The ROI average fit is plot against the fit of all of the others for comparison.

    Note here and elsewhere the single pixel check is the first element of the AOI array.

    Arguments
    ---------
    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    ref : np array, 3D
        Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    sweep_list : list
        List of sweep parameter values (with removed unwanted sweeps at start/end)
    fit_result_lst : list
        List of `dukit.pl.common.FitResultCollection` objects (one for each fit_backend)
        holding ROI, AOI fit results
    fit_model : dukit.pl.model.FitModel
        Model we're fitting to.
    opath : str
        If given, tries to save figure here.
    norm : str
    *aoi_coords : tuple of tuple[int, int, int, int]

    Returns
    -------
    fig : matplotlib Figure object
    """

    # rows:
    # ROI avg, single pixel, then each AOI
    # columns:
    # sig & ref, sub & div norm, fit -> compared to ROI {raw, fit, ROI_avg_fit}

    aois = _get_aois(sig.shape, *aoi_coords)  # type: ignore

    figsize = mpl.rcParams["figure.figsize"].copy()
    figsize[0] *= 3  # number of columns
    figsize[1] *= 2 + len(aois)  # number of rows

    fig, axs = plt.subplots(
            2 + len(aois), 3, figsize=figsize, sharex=True, sharey=False
    )

    #  pre-process raw data to plot -> note some are not averaged yet
    #  (will check for this below)
    sigs = []
    refs = []
    sigs.append(sig)
    refs.append(ref)
    # add AOI data
    for i, aoi in enumerate(aois):
        aoi_sig = sig[:, aoi[0], aoi[1]]
        aoi_ref = ref[:, aoi[0], aoi[1]]
        sigs.append(aoi_sig)
        refs.append(aoi_ref)

    # plot sig, ref data as first column
    for i, (s, r) in enumerate(zip(sigs, refs)):
        if len(s.shape) > 1:
            s_avg = np.nanmean(s, axis=(1, 2))
            r_avg = np.nanmean(r, axis=(1, 2))
        else:
            s_avg = s
            r_avg = r
        # plot sig
        axs[i, 0].plot(
                sweep_list,
                s_avg,
                label="sig",
                c="blue",
                ls="dashed",
                marker="o",
                mfc="cornflowerblue",
                mec="mediumblue",
        )
        # plot ref
        axs[i, 0].plot(
                sweep_list,
                r_avg,
                label="ref",
                c="green",
                ls="dashed",
                marker="o",
                mfc="limegreen",
                mec="darkgreen",
        )

        axs[i, 0].legend()
        axs[i, 0].grid(True)
        if not i:
            axs[i, 0].set_title("ROI avg")
        elif i == 1:
            axs[i, 0].set_title(
                    "Single Pixel Check",
                    fontdict={"color": AOI_COLORS[0]},
            )
        else:
            axs[i, 0].set_title(
                    "AOI " + str(i - 1) + " avg",
                    fontdict={"color": AOI_COLORS[i - 1]},
            )
        axs[i, 0].set_ylabel("pl (a.u.)")
    axs[-1, 0].set_xlabel("Sweep parameter")

    # plot normalisation as second column
    for i, (s, r) in enumerate(zip(sigs, refs)):
        if len(s.shape) > 1:
            sub_avg = np.nanmean(1 + (s - r) / (s + r), axis=(1, 2))
            div_avg = np.nanmean(s / r, axis=(1, 2))
        else:
            sub_avg = 1 + (s - r) / (s + r)
            div_avg = s / r

        axs[i, 1].plot(
                sweep_list,
                sub_avg,
                label="subtraction",
                c="firebrick",
                ls="dashed",
                marker="o",
                mfc="lightcoral",
                mec="maroon",
        )
        axs[i, 1].plot(
                sweep_list,
                div_avg,
                label="division",
                c="cadetblue",
                ls="dashed",
                marker="o",
                mfc="powderblue",
                mec="darkslategrey",
        )

        axs[i, 1].legend()
        axs[i, 1].grid(True)
        if not i:
            axs[i, 1].set_title("ROI avg - Normalisation")
        elif i == 1:
            axs[i, 1].set_title(
                    "Single Pixel Check - Normalisation",
                    fontdict={"color": AOI_COLORS[0]},
            )
        else:
            axs[i, 1].set_title(
                    "AOI " + str(i - 1) + " avg - Normalisation",
                    fontdict={"color": AOI_COLORS[i - 1]},
            )
        axs[i, 1].set_ylabel("pl (a.u.)")
    axs[-1, 1].set_xlabel(
            "Sweep parameter"
    )  # this is meant to be less indented than the line above

    high_res_xdata = np.linspace(
            np.min(fit_result_lst[0].ROI_avg_fit_result.sweep_list),
            np.max(fit_result_lst[0].ROI_avg_fit_result.sweep_list),
            10000,
    )

    # loop of fit backends first
    for fit_backend_number, fit_backend_fit_result in enumerate(
            fit_result_lst
    ):
        fit_backend_name = fit_backend_fit_result.fit_backend

        fit_params_lst = [
            fit_backend_fit_result.ROI_avg_fit_result.best_params,
            fit_backend_fit_result.single_pixel_fit_result,
            *fit_backend_fit_result.AOI_fit_results_lst,
        ]
        # now plot fits as third column
        for i, (fit_param_ar, s, r) in enumerate(zip(fit_params_lst, sigs, refs)):
            if norm == "div":
                sig_norm = s / r
            elif norm == "sub":
                sig_norm = 1 + (s - r) / (s + r)
            elif norm == "true_sub":
                sig_norm = (s - r) / np.nanmax(s - r)

            sig_norm_avg = (
                np.nanmean(sig_norm, axis=(1, 2))
                if len(sig_norm.shape) > 1
                else sig_norm
            )

            best_fit_ydata = fit_model(fit_param_ar, high_res_xdata)
            roi_fit_ydata = fit_model(
                    fit_backend_fit_result.ROI_avg_fit_result.best_params,
                    high_res_xdata,
            )

            # this is the first loop -> plot raw data, add titles
            if not fit_backend_number:
                # raw data
                axs[i, 2].plot(
                        sweep_list,
                        sig_norm_avg,
                        label="raw data",
                        ls="",
                        marker="o",
                        ms=3.5,
                        mfc="goldenrod",
                        mec="k",
                )
                if not i:
                    axs[i, 2].set_title("ROI avg - Fit")
                elif i == 1:
                    axs[i, 2].set_title(
                            "Single Pixel Check - Fit",
                            fontdict={"color": AOI_COLORS[0]},
                    )
                else:
                    axs[i, 2].set_title(
                            "AOI " + str(i - 1) + " avg - Fit",
                            fontdict={"color": AOI_COLORS[i - 1]},
                    )
            # ROI avg fit (as comparison)
            if i:
                axs[i, 2].plot(
                        high_res_xdata,
                        roi_fit_ydata,
                        label=f"ROI avg fit - {fit_backend_name}",
                        ls="dashed",
                        c=FIT_BACKEND_COLORS[fit_backend_name]["AOI_ROI_fit_linecolor"],
                )
            # best fit
            axs[i, 2].plot(
                    high_res_xdata,
                    best_fit_ydata,
                    label=f"fit - {fit_backend_name}",
                    ls="dashed",
                    c=FIT_BACKEND_COLORS[fit_backend_name]["AOI_best_fit_linecolor"],
            )

            axs[i, 2].legend()
            axs[i, 2].grid(True)
            axs[i, 2].set_ylabel("pl (a.u.)")

    axs[-1, 2].set_xlabel(
            "Sweep parameter"
    )  # this is meant to be less indented than line above

    if opath:
        fig.savefig(opath)

    return fig


# ============================================================================


def pl_param_image(
        fit_model,
        pixel_fit_params: dict,
        param_name: str,
        param_number: int = 0,
        errorplot: bool = False,
        opath: str = "",
        **kwargs,
):
    """
    Plots an image corresponding to a single parameter in pixel_fit_params.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    fit_model : dukit.pl.model.FitModel
        Model we're fitting to.
    pixel_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    param_name : str
        Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'.

    Optional arguments
    ------------------
    param_number : int, default-0
        Which version of the parameter you want. I.e. there might be 8 independent
         parameters in the fit model called 'pos', each labeled 'pos_0', 'pos_1' etc.
    errorplot : bool
        Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so
        ylabel & save names are changed accordingly.
        Can't be True if param_name='residual'.
    opath : str
        If given, tries to save figure here.
    **kwargs
        Plotting options passed to `dukit.shared.plot.plot_image_on_ax`

    Returns
    -------
    fig : matplotlib Figure object
    """

    image = pixel_fit_params[param_name + "_" + str(param_number)]
    if param_name == "residual" and errorplot:
        dukit.shared.misc.dukit_warn(
                "residual doesn't have an error, can't plot residual sigma (ret. None)."
        )
        return None

    if errorplot:
        c_label = "SD: " + fit_model.get_param_unit(param_name, param_number)
    else:
        c_label = fit_model.get_param_unit(param_name, param_number)

    if "c_range_type" in kwargs and "c_range_vals" in kwargs:
        c_range = dukit.shared.plot.get_colormap_range(
                kwargs["c_range_type"], kwargs["c_range_vals"], image
        )
    else:
        c_range = dukit.shared.plot.get_colormap_range("min_max", (), image)

    fig, ax = plt.subplots()
    fig, ax = dukit.shared.plot.plot_image_on_ax(
            fig,
            ax,
            image,
            title=param_name + "_" + str(param_number),
            c_range=c_range,
            c_label=c_label,
            **kwargs,
    )

    if opath:
        fig.savefig(opath)

    return fig, ax


# ============================================================================


# TODO add fit_model type
def pl_param_images(
        fit_model,
        pixel_fit_params: dict,
        param_name: str,
        errorplot: bool = False,
        opath: str = "",
        **kwargs,
):
    """
    Plots images for all independent versions of a single parameter type in
    pixel_fit_params.

    Arguments
    ---------
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fitting to.
    pixel_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    param_name : str
        Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'.
    errorplot : bool
        Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so
        ylabel & save names are changed accordingly.
        Can't be True if param_name='residual'.
    opath : str
        If given, tries to save figure here.
    **kwargs
        Plotting options passed to `dukit.shared.plot.plot_image_on_ax`

    Returns
    -------
    fig : plt.Figure

    ax : plt.Axes
    """

    # if no fit completed
    if pixel_fit_params is None:
        dukit.shared.misc.dukit_warn(
                "'pixel_fit_params' arg to function 'pl_param_images' is 'None'.\n"
                + "Probably no pixel fitting completed."  # noqa: W503
        )
        return None

    if param_name == "residual" and errorplot:
        dukit.shared.misc.dukit_warn(
                "residual doesn't have an error, can't plot residual sigma (ret. None)."
        )
        return None

    # plot 2 columns wide, as many rows as required

    # first get keys we need
    our_keys = []
    for key in pixel_fit_params:
        if key.startswith(param_name):
            our_keys.append(key)

    # this is an inner function so no one uses it elsewhere/protect namespace
    def param_sorter(param):
        strings = param.split("_")  # e.g. "amp_exp_2" -> ["amp", "exp", "2"]
        # all we need here is the number at the end actually
        # param_name_split = strings[:-1]  # list of 'words', e.g. ["amp", "exp"]
        num = strings[-1]  # grab the number e.g. "2
        return int(num)

    # sort based on number (just in case)
    our_keys.sort(key=param_sorter)  # i.e. key = lambda x: int(x.split("_")[-1])
    nk = len(our_keys)

    if nk == 1:
        # just one image, so plot normally
        fig = pl_param_image(fit_model, pixel_fit_params, param_name, 0, errorplot)
    else:
        if nk <= 8:
            num_columns = 4
            num_rows = 2
        else:
            num_columns = 2
            num_rows = math.ceil(nk / 2)

        figsize = mpl.rcParams["figure.figsize"].copy()
        figsize[0] *= num_columns
        figsize[1] *= num_rows

        fig, axs = plt.subplots(
                num_rows,
                num_columns,
                figsize=figsize,
                sharex=False,
                sharey=False,
        )
        # plot 8-lorentzian peaks in a more helpful way
        if nk <= 8 and any(
                [f.startswith("lorentzian") for f in fit_model.fit_functions]
        ):
            param_nums = []  # [0, 1, 2, 3, 7, 6, 5, 4] etc.
            param_nums.extend(list(range(nk // 2)))
            if nk % 2:
                param_nums.append(nk // 2 + 1)
            if len(param_nums) < 4:
                param_nums.extend([-1 for _ in range(4 - len(param_nums))])  # dummies
            param_nums.extend(
                    list(range(nk - 1, (nk - 1) // 2, -1))
            )  # range(start, stop, step)
            param_nums.extend(
                    [-1 for _ in range(8 - len(param_nums))]
            )  # add on dummies
            param_axis_iterator = zip(param_nums, axs.flatten())
        # otherwise plot in a more conventional order
        else:
            param_axis_iterator = enumerate(axs.flatten())  # type: ignore

        for param_number, ax in param_axis_iterator:
            param_key = param_name + "_" + str(param_number)
            try:
                image_data = pixel_fit_params[param_key]
            except KeyError:
                # we have too many axes (i.e. 7 params, 8 subplots), delete the axs
                fig.delaxes(ax)
                continue

            if errorplot:
                c_label = "SD: " + fit_model.get_param_unit(param_name, param_number)
            else:
                c_label = fit_model.get_param_unit(param_name, param_number)

            if "c_range_type" in kwargs and "c_range_vals" in kwargs:
                c_range = dukit.shared.plot.get_colormap_range(
                        kwargs["c_range_type"], kwargs["c_range_vals"], image_data
                )
            else:
                c_range = dukit.shared.plot.get_colormap_range(
                        "min_max", (), image_data
                )

            fig, ax = dukit.shared.plot.plot_image_on_ax(
                    fig,
                    ax,
                    image_data,
                    title=param_key,
                    c_range=c_range,
                    c_label=c_label,
                    **kwargs,
            )

        if opath:
            fig.savefig(opath)

    return fig, ax


# ============================================================================


def _add_patch_rect(
        ax: plt.Axes,
        aoi_coord: tuple[int, int, int, int],
        label: str | None = None,
        edgecolor: str = "b",
) -> None:
    """
    Adds a rectangular annotation onto ax.

    Arguments
    ---------
    ax : matplotlib Axis object

    aoi_coord: tuple[int, int, int, int]
        start_x, start_y, end_x, end_y for aoi/roi

    Optional arguments
    ------------------
    label : str
        Text to label annotated square with. Color is defined by edgecolor. Default: None.
    edgecolor : str
        Color of label and edge of annotation. Default: "b".
    """
    start_x, start_y, end_x, end_y = aoi_coord
    rect = patches.Rectangle(
            (start_x, start_y),
            int(end_x - start_x),
            int(end_y - start_y),
            linewidth=1,
            edgecolor=edgecolor,
            facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        ax.text(
                start_x + 0.95 * int(end_x - start_x),  # label posn.: top right
                start_y,
                label,
                {
                    "color": edgecolor,
                    "fontsize": 10,
                    "ha": "center",
                    "va": "bottom",
                },
        )


# ============================================================================


def _get_aois(
        image_shape: tuple[int, int, int] | tuple[int, int],
        *aoi_coords: tuple[int, int, int, int],
) -> list[tuple[npt.NDArray, npt.NDArray]]:
    aois: list = (
        []
        if not aoi_coords
        else [dukit.shared.misc._define_area_roi(*aoi) for aoi in aoi_coords]
    )

    if len(image_shape) == 3:
        shp = image_shape[1:]
    else:
        shp = image_shape
    aois.insert(
            0,
            dukit.shared.misc._define_area_roi(
                    shp[0] // 2, shp[1] // 2, shp[0] // 2 + 1, shp[1] // 2 + 1
            ),
    )
    return aois
