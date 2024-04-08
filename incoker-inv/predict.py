"""
This module is designed to predict material properties based on microstructure features using pre-trained Gaussian
Process Regression (GPR) models. It loads a specified model and its associated features, extracts the necessary
features from the provided microstructure data, and uses the model to predict the material property.

Usage:
    Run this script from the command line, providing the microstructure information in JSON format as an argument.
    Example:
        python predict.py '{"feature1": value1, "feature2": value2, ...}' --model_file <path_to_surrogate_model.joblib>
"""

import argparse

import joblib
import numpy as np
import json


def main(microstructure, model_file):
    """
    Predict the property of a material based on its microstructure features.

    Args:
        microstructure (dict): A dictionary containing the microstructure features as key-value pairs.
        model_file (str): A string containing the path to the surrogate GPR model.

    Returns:
        float: The predicted thermal conductivity value.
    """
    models = joblib.load(model_file)["models"]
    model = models["thermal_conductivity"]["pipe"]
    features = models["thermal_conductivity"]["features"]

    # Extract the features required by the specific property's GPR model
    microstructure_features = [microstructure[feature] for feature in features]
    # Prepare the input for prediction
    X = np.array(microstructure_features).reshape(1, -1)
    # Use the GPR model to predict the property value
    predicted_value = model.predict(X)
    print(predicted_value)
    return predicted_value[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict property")
    parser.add_argument("prop", type=json.loads, help="Expected property in JSON format")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the model file")
    args = parser.parse_args()

    main(args.prop, args.model_file)
