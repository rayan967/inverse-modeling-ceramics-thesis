"""Script for cleaning up directories for restarting failed runs of online inverse design training."""

import json
import pathlib
import shutil

input_folder = pathlib.Path("adaptive_points")

rve_roots = list(input_folder.glob("*"))

for rve_root in rve_roots:
    simulation_result_files = list(rve_root.glob("**/results__thermal_conductivity.json"))
    info_file = pathlib.Path(rve_root, "info.json")

    if len(simulation_result_files) == 0 or not info_file.exists():
        # no simulation result present -> delete parent folder
        shutil.rmtree(rve_root)
        continue

    # update info.json file with data from simulation result for successful simulation runs
    rve_info = json.loads(info_file.read_text())
    sim_result = json.loads(simulation_result_files[0].read_text())

    cl_11 = rve_info["chord_length_analysis"]["phase_chord_lengths"]["11"]["mean_chord_length"]
    cl_4 = rve_info["chord_length_analysis"]["phase_chord_lengths"]["4"]["mean_chord_length"]
    clr = cl_11 / cl_4
    rve_info["chord_length_ratio"] = clr

    rve_info.update(sim_result)

    info_file.write_text(json.dumps(rve_info, indent=4))
