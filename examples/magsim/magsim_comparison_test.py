import dukit
import os
import numpy as np
import matplotlib.pyplot as plt

with open(
    os.path.dirname(__file__) + "/../TEST_DATA_PATH.py", encoding="utf-8"
) as fid:
    exec(fid.read())  # reads in TEST_DATA_PATH string
DIR = TEST_DATA_PATH + "mz_test/"

np_text_file_path = (
    DIR
    + "ODMR - Pulsed_10_Rectangle_bin_8/field/sig_sub_ref/sig_sub_ref_bnv_0.txt"
)
json_output_path = DIR + "polys_mz_comparison.json"
json_input_path = DIR + "polys.json"
mean_plus_minus = 0.25

pgon_patch = {
    "facecolor": None,
    "edgecolor": "xkcd:grass green",
    "linestyle": "dashed",
    "fill": False,
    "linewidth": 2,
}

mesh_size = 512
height = 290e-9
res = 700e-9
fov_size = 30e-6

sim = dukit.ComparisonMagSim(np_text_file_path, (30e-6, 30e-6))

# add from prev. file
# sim.add_polygons(json_input_path)
# OR gui-select polygons, use 'alt' to signal finished polygon (can have multiple)
# (shift + mouse to move, r + scroll to resize, for example), close window to continue
sim.select_polygons(
    output_path=json_output_path, mean_plus_minus=mean_plus_minus
)

sim.rescale(3)  # to resize polygons

sim.define_magnets(5, (0, 0, 1))

sim.plot_magsim_magnetizations(
    annotate_polygons=True, polygon_patch_params=pgon_patch
)
sim.run(290e-9, pad_mode="constant", resolution=700e-9)
unv = [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]

sim.plot_magsim_bfield_at_nvs(
    strict_range=(-0.25, 0.25), projection=unv
)  # these return fig, ax
sim.plot_comparison(
    strict_range=(-0.25, 0.25), projection=unv
)  # so you could e.g. run: fig, _ = sim.plot_comparison(); fig.savefig(path)

plt.show()
