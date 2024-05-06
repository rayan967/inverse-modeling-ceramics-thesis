"""Script to generate validation data."""

import pathlib
import shutil

import numpy as np
from incoker_micro_sims.prediction_pipeline import generate_and_predict

from incoker_inverse.online_inverse_design.training.generate_predict_utils import (
    property_dict_category,
)
from incoker_inverse.online_inverse_design.training.online_training import load_config
from incoker_inverse.simlopt.basicfunctions.utils.creategrid import createPD

current_file = pathlib.Path(__file__).resolve()
file_directory = current_file.parent
config_path = file_directory / "validation_config.yaml"
# Load the configuration file
config = load_config(config_path)

property_name = config["property_name"]
simulation_options = config["simulation_options"]
simulation_options["material_property"] = property_dict_category[property_name]
parameterranges = np.array(
    [config["parameterranges"]["VolumeFractionZirconia"], config["parameterranges"]["ChordLengthRatio"]]
)

base_path = pathlib.Path(config["validation_data_path"] + "_" + simulation_options["material_property"])
NMC = config["grid_size"]
base_path.mkdir(parents=True, exist_ok=True)

XGLEE = createPD(NMC, parameterranges.shape[0], "grid", parameterranges, [])

for input in XGLEE:
    output_path = pathlib.Path(base_path, f"v={input[0]:.2f},r={input[1]:.2f}")
    simulation_options["output_path"] = output_path
    try:
        result = generate_and_predict(input, simulation_options)
    except Exception:
        print("error -> removing folder")
        shutil.rmtree(output_path)
        continue

# Delete the remaining files except info.json
# Loop through each subdirectory in the base path
for subdir in base_path.iterdir():
    if subdir.is_dir():  # Ensure it's a directory
        # Define the path to the 'rve_0' subdirectory
        rve_path = subdir / "rve_0"

        # Define the source and target path for 'info.json'
        source_json = rve_path / "info.json"
        target_json = subdir / "info.json"

        # Move 'info.json' to the parent directory
        if source_json.exists():
            shutil.move(str(source_json), str(target_json))

        # Remove the 'rve_0' directory if it's empty or if you want to force delete
        if rve_path.exists():
            shutil.rmtree(rve_path)
pass
