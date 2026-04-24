from __future__ import annotations

import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_MODULES = [
    "scripts.run_rafaely_ch2_planewave_freefield_sphere",
    "scripts.run_rafaely_ch2_planewave_freefield_xy",
    "scripts.run_rafaely_ch2_planewave_rigid_sphere",
    "scripts.run_rafaely_ch2_planewave_rigid_xy",
    "scripts.run_rafaely_ch2_radial_functions_1",
    "scripts.run_rafaely_ch2_radial_functions_2",
    "scripts.run_rafaely_ch3_gaussian",
    "scripts.run_rafaely_ch3_aliasing_example",
    "scripts.run_rafaely_ch3_aliasing_matrix",
    "scripts.run_rafaely_ch3_platonic_solids",
    "scripts.run_rafaely_ch3_sampling_schemes",
    "scripts.run_rafaely_ch4_array_condition_numbers",
    "scripts.run_rafaely_ch4_array_design_examples",
    "scripts.run_rafaely_ch4_array_radial_functions",
    "scripts.run_rafaely_ch4_cardioid_directivity",
    "scripts.run_rafaely_ch5_wng_example",
    "scripts.run_rafaely_ch5_beamforming_example",
    "scripts.run_rafaely_ch5_omni_and_directional",
    "scripts.run_rafaely_ch7_lcmv_beampatterns_1",
    "scripts.run_rafaely_ch7_lcmv_beampatterns_2",
    "scripts.run_rafaely_ch7_mvdr_beampatterns_1",
    "scripts.run_rafaely_ch7_mvdr_beampatterns_2",
]


def test_run_missing_chapters_smoke():
    for mod_name in RUN_MODULES:
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "main")
        mod.main(show=False)
        assert len(plt.get_fignums()) > 0
        plt.close("all")
