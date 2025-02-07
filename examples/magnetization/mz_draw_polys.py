import dukit
import matplotlib.pyplot as plt
import os

with open(
    os.path.dirname(str(__file__)) + "/../TEST_DATA_PATH.py", encoding="utf-8"
) as fid:
    exec(fid.read())  # reads in TEST_DATA_PATH string

DIR = TEST_DATA_PATH + "mz_test/"  # type: ignore
FILEPATH = DIR + "ODMR - Pulsed_10"

numpy_txt_file_path = FILEPATH + "_output/data/b_nv_sbg_0.txt"

# json_input_path = TEST_DATA_PATH + "..."
json_output_path = FILEPATH + "_new_polys.json"

mean_plus_minus = 5e-6

pgon_lst = dukit.polygon_selector(
    numpy_txt_file_path,
    # json_input_path=json_input_path,
    json_output_path=json_output_path,
    mean_plus_minus=mean_plus_minus,
)
