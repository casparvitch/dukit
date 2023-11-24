# -*- coding: utf-8 -*-
"""
This module holds tools for measurement (lateral) drift correction.
Requires a `dukit.shared.systems.System` for reading files from disk.
Most of the functions are not documented, but the API is only 2 funcs:

Functions
---------
 - `dukit.driftcorrect.drift_correct_test`
 - `dukit.driftcorrect.drift_correct_measurement`

"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.driftcorrect.drift_correct_test": True,
    "dukit.driftcorrect.drift_correct_measurement": True,
}

# ============================================================================

from typing import Callable
import numpy as np
import numpy.typing as npt
from tqdm.autonotebook import tqdm
from skimage.registration import phase_cross_correlation as cross_correl
from skimage.transform import EuclideanTransform, warp
import matplotlib.pyplot as plt

# ============================================================================

import dukit.shared.plot
import dukit.shared.misc
import dukit.shared.systems

# ============================================================================


def _drift_correct_single(
    refr_pl: npt.NDArray, move_pl: npt.NDArray, target_pl: npt.NDArray
) -> tuple[npt.NDArray, tuple[int, int]]:
    # refr_pl & move_pl should probably be cropped to some feature
    # then the shift is applied to target_pl
    with np.errstate(all="ignore"):
        shift_calc = cross_correl(refr_pl, move_pl, normalization=None)[0]
    tform = EuclideanTransform(translation=[-shift_calc[1], -shift_calc[0]])
    return warp(target_pl, tform, mode="edge"), tuple(shift_calc)


def _drift_correct_stack(
    refr_pl: npt.NDArray, move_pl: npt.NDArray, move_sig_norm: npt.NDArray
) -> tuple[npt.NDArray, tuple[int, int]]:
    # calc's shift in 'mov_pl' to match 'refr_pl' (image regulation) via
    # cross-corr method then applies that to move_sig_norm.
    with np.errstate(all="ignore"):
        shift_calc = cross_correl(refr_pl, move_pl, normalization=None)[0]
    tform = EuclideanTransform(translation=[-shift_calc[1], -shift_calc[0]])

    reg_sig_norm = np.empty(move_sig_norm.shape)
    for i in range(move_sig_norm.shape[0]):
        reg_sig_norm[i, :, :] = warp(move_sig_norm[i, :, :], tform, mode="edge")
    return reg_sig_norm, tuple(shift_calc)


# ============================================================================


def read_and_drift_correct(
    base_path: str,
    stub: Callable[[int], str],
    image_seq: list | tuple | npt.NDArray,
    system: dukit.shared.systems.System,
    ignore_ref: bool = False,
    mask: npt.NDArray[np.bool_]
    | None = None,  # True where(i) you want to incl im in accum
    roi_start_x: int = -1,
    roi_start_y: int = -1,
    roi_end_x: int = -1,
    roi_end_y: int = -1,
) -> tuple[npt.NDArray, npt.NDArray]:
    roi = roi_start_x, roi_start_y, roi_end_x, roi_end_y
    first = True
    for i in tqdm(image_seq):
        sig, ref, _ = system.read_image(base_path + stub(i), ignore_ref, "div")
        if first:
            first = False
            refr_pl = np.sum(sig, axis=0)
            accum_sig = sig.copy()
            accum_ref = ref.copy()
            prev_sig = sig.copy()
            prev_ref = ref.copy()
            continue

        this_sig = sig - prev_sig
        this_ref = ref - prev_ref
        this_pl = np.sum(this_sig, axis=0)
        prev_sig = sig
        prev_ref = ref

        this_sig, _ = _drift_correct_stack(
            dukit.shared.misc.crop_roi(refr_pl, *roi),
            dukit.shared.misc.crop_roi(this_pl, *roi),
            this_sig,
        )
        this_ref, _ = _drift_correct_stack(
            dukit.shared.misc.crop_roi(refr_pl, *roi),
            dukit.shared.misc.crop_roi(this_pl, *roi),
            this_ref,
        )

        if mask is None or mask[i - image_seq[0]]:
            accum_sig += this_sig
            accum_ref += this_ref

    return accum_sig, accum_ref


# ============================================================================


def drift_correct_measurement(
    directory: str,
    start_num: int,
    end_num: int,
    stub: Callable[[int], str],
    system: dukit.shared.systems.System,
    output_file: str,
    roi_start_x: int = -1,
    roi_start_y: int = -1,
    roi_end_x: int = -1,
    roi_end_y: int = -1,
    image_nums_mask=None,
):
    """

    Arguments
    ---------
    directory : str
        Path to directory that contains all the measurements.
    start_num : int
        Image/measurement to start accumulating from.
    end_num : int
        Image/measurement to end accumulation.
    stub : Callable[[int], str]
        Function that takes image num and returns path to that measurement
        from directory. I.e. directory + stub(X) for filepath ass. with 'X'
    output_file : str
        Output will be stored in directory + output file
    system : dukit.shared.systems.System object
        Used for reading in files
    output_file : str
        Where to save the drift-corrected binary
        Format is based on 'system'.
    roi_X : int
        Define roi, here a feature in PL that is used from cross-correlation.
    image_nums_mask : 1D ndarray
        Same shape as list(range(start_num, end_num + 1)).
        Where false, don't include that image in accumulation.
    """

    sig, ref = read_and_drift_correct(
        directory,
        stub,
        list(range(start_num, end_num + 1)),
        system,
        roi_start_x=roi_start_x,
        roi_start_y=roi_start_y,
        roi_end_x=roi_end_x,
        roi_end_y=roi_end_y,
        mask=image_nums_mask,
    )

    if isinstance(system, dukit.shared.systems.LVControl):
        s = sig.transpose([0, 2, 1])
        r = ref.transpose([0, 2, 1])

        output = []
        for f in range(s.shape[0]):
            output.append(s[f, ::])
            output.append(r[f, ::])
        output_array = np.array(output).flatten()

        with open(output_file, "wb") as fid:
            np.array([0, 0]).astype(np.float32).tofile(fid)
            output_array.astype(np.float32).tofile(fid)

        lines = []
        with open(directory + stub(start_num) + "_metaSpool.txt", "r") as fid:
            for line in fid:
                # assume no binning eh
                # if line.startswith("Binning:"):
                #     old_binning = int(re.match(r"Binning: (\d+)\n", line)[1])
                #     new_binning = (
                #         old_binning
                #         if not additional_bins
                #         else old_binning * additional_bins
                #     )
                #     lines.append(f"Binning: {new_binning}\n")
                # else:
                lines.append(line)

        with open(output_file + "_metaSpool.txt", "w") as fid:
            for line in lines:
                fid.write(line)
    else:
        raise ValueError("Don't know how to write that type of binary yet.")


# ============================================================================


def drift_correct_test(
    directory: str,
    start_num: int,
    end_num: int,
    comparison_nums: list | tuple | npt.NDArray,
    stub: Callable[[int], str],
    system: dukit.shared.systems.System,
    ignore_ref:bool=False,
    roi_start_x: int = -1,
    roi_start_y: int = -1,
    roi_end_x: int = -1,
    roi_end_y: int = -1,
):
    """
    Test the drift correction on a subset (comparison_nums) of the measurements.

    Arguments
    ---------
    directory : str
        Path to directory that contains all the measurements.
    start_num : int
        Image/measurement to start accumulating from.
        (Also the reference frame)
    end_num : int
        Image/measurement to end accumulation.
    comparison_nums : list of ints
        List of image/measurment nums to compare drift calc on.
    stub : function
        Function that takes image num and returns path to that measurement
        from directory. I.e. directory + stub(X) for filepath ass. with 'X'
    system : dukit.shared.systems.System object
        Used for reading in files
    ignore_ref : bool = False
    roi_X : int
        Define roi, here a feature in PL that is used from cross-correlation.

    Returns
    -------
    crop_fig : plt.Figure
        For further editing/saving etc.
    crop_axs : plt.Axes
        For further editing/saving etc.
    """

    # prep fig
    nrows = len(comparison_nums)
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 4 * nrows))
    roi = roi_start_x, roi_start_y, roi_end_x, roi_end_y

    # read in all frames, only pull out (non-accum) pl frames we want to compare
    raw_comp_frames = []
    image_seq = list(range(start_num, end_num + 1))
    first = True
    for i in image_seq:
        sig, _, _ = system.read_image(directory + stub(i), ignore_ref, "div")
        pl = np.sum(np.sum(sig, axis=-1), axis=-1)
        accum_pl = dukit.shared.misc.crop_roi(pl, *roi)
        if first:
            first = False
            prev_accum_pl = accum_pl.copy()
            if i in comparison_nums:
                raw_comp_frames.append(accum_pl)
            continue
        this_pl = accum_pl - prev_accum_pl
        prev_accum_pl = accum_pl
        if i in comparison_nums:
            raw_comp_frames.append(this_pl)
        if i > max(comparison_nums):
            break

    # plot cropped sig frames in left column
    for i, frame, ax in zip(comparison_nums, raw_comp_frames, axs[:, 0]):
        dukit.shared.plot.plot_image_on_ax(
            fig,
            ax,
            dukit.shared.misc.crop_roi(frame, *roi),
            title=f"raw   {i}",
            c_range=(None, None),
            c_label="Counts",
            c_map="gray",
        )

    # do cross-corr on comparison frames
    refr_frame = raw_comp_frames[0]
    corrected_frames = [
        refr_frame,
    ]
    shift_calcs: list[tuple[int, int]] = [
        (0, 0),
    ]
    for frame in raw_comp_frames[1:]:
        corrected_frame, shift_calc = _drift_correct_single(
            dukit.shared.misc.crop_roi(refr_frame, *roi),
            dukit.shared.misc.crop_roi(frame, *roi),
            frame,
        )
        corrected_frames.append(corrected_frame)
        shift_calcs.append(shift_calc)

    # plot cropped corrected frames in right column
    for i, frame, shift_calc, ax in zip(
        comparison_nums, corrected_frames, shift_calcs, axs[:, 1]
    ):
        dukit.shared.plot.plot_image_on_ax(
            fig,
            ax,
            dukit.shared.misc.crop_roi(frame, *roi),
            title=f"shftd {i}: {shift_calc}",
            c_range=(None, None),
            c_label="Counts",
        )

    return fig, axs
