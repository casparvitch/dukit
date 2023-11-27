import dukit
import os
from os.path import dirname
import matplotlib.pyplot as plt

with open(dirname(str(__file__)) + "/../TEST_DATA_PATH.py", encoding="utf-8") as fid:
    exec(fid.read())  # reads in TEST_DATA_PATH string

# === SET PARAMS

DIR = TEST_DATA_PATH + "mz_test/"  # type: ignore
FILEPATH = DIR + "ODMR - Pulsed_10"
ADDITIONAL_BINS = 4
ADDITIONAL_SMOOTH = 0.0
ROI_COORDS = (65, 65, 190, 190)  # start_x, start_y, ...
AOI_COORDS = ((30, 40, 40, 45), (20, 20, 24, 24), (50, 50, 51, 51))
# AOI_COORDS = ((30, 40, 40, 45),)

# === CREATE OUTPUT DIR & set mpl rcparams
OUTPUT_DIR = f"{FILEPATH}_output/"
try:
    os.mkdir(OUTPUT_DIR)
except FileExistsError:
    pass
dukit.mpl_set_run_config()

# === READ IN DATA
sys = dukit.CryoWidefield()
sweep_arr = sys.read_sweep_arr(FILEPATH)
sig, ref, sig_norm = sys.read_image(FILEPATH)

# === REBIN & CROP
sig_rebinned = dukit.rebin_image_stack(sig, ADDITIONAL_BINS)
pl_img = dukit.sum_spatially(sig_rebinned)
sig = dukit.crop_roi(sig_rebinned, ROI_COORDS)
del sig_rebinned  # hacky but I want to keep the memory clear
ref = dukit.crop_roi(dukit.rebin_image_stack(ref, ADDITIONAL_BINS), ROI_COORDS)
sig_norm = dukit.crop_roi(
        dukit.rebin_image_stack(sig_norm, ADDITIONAL_BINS), ROI_COORDS
)
pl_img_crop = dukit.sum_spatially(sig)

# === PLOT PL INFO
_ = dukit.plot.roi_pl_image(
        pl_img,
        ROI_COORDS,
        opath=OUTPUT_DIR + "pl_full.svg",
        show_tick_marks=True,
)
_ = dukit.plot.aoi_pl_image(
        pl_img_crop,
        *AOI_COORDS,
        opath=OUTPUT_DIR + "pl_full.svg",
        show_tick_marks=True,
)
_ = dukit.plot.aoi_spectra(
        sig,
        ref,
        sweep_arr,
        specpath=OUTPUT_DIR + "aoi_specta.json",
        opath=OUTPUT_DIR + "aoi_spectra.svg",
        *AOI_COORDS,
)

plt.show()
