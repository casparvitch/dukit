import dukit
import os
import numpy as np
import matplotlib.pyplot as plt

with open(
    os.path.dirname(__file__) + "/../TEST_DATA_PATH.py", encoding="utf-8"
) as fid:
    exec(fid.read())  # reads in TEST_DATA_PATH string
DIR = TEST_DATA_PATH + "mz_test/"

np_text_file_path = DIR + "ODMR - Pulsed_10_Rectangle_bin_8/field/sig_sub_ref/sig_sub_ref_bnv_0.txt"
json_output_path = DIR + "polys_mz_sandbox.json"
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

sim = dukit.SandboxMagSim((mesh_size, mesh_size), (fov_size, fov_size))

# import some polygons as a template
sim.add_template_polygons(json_input_path)

# now adjust the template (shift + mouse to move, r + scroll to resize, for example)
# close window to continue
sim.adjust_template(output_path=json_output_path)
sim.set_template_as_polygons()

sim.set_template_as_polygons()

sim.define_magnets(5,  (0, 1, 0))  # mag unit: mu_b/nm^2
_ = sim.plot_magsim_magnetizations(annotate_polygons=True, polygon_patch_params=pgon_patch)

sim.run(
    height, pad_mode="constant", resolution=res
)  # height: 'PX' equivalent in z, res the same

unv = [np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3)]

_ = sim.plot_magsim_bfield_at_nvs(strict_range=(-0.25, 0.25), projection=unv)

_ = sim.plot_magsim_bfield_at_nvs(strict_range=(-0.25, 0.25), projection=(1, 0, 0))
_ = sim.plot_magsim_bfield_at_nvs(strict_range=(-0.25, 0.25), projection=(0, 1, 0))
_ = sim.plot_magsim_bfield_at_nvs(strict_range=(-0.25, 0.25), projection=(0, 0, 1))

plt.show()