"""Visualize the evolution of the GP model through the adaptive phases."""

import json
import pathlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib import cm
from sklearn import metrics


def load_test_data(base_path):
    """
    Load test data from JSON files within a specified directory.

    Parameters:
    - base_path (pathlib.Path): Directory path to load data from.
    - prop_name (str): Material property considered.

    Returns:
    - tuple: A tuple containing two numpy arrays, X (features) and y (target values).
    """
    base_path = pathlib.Path(base_path)
    info_files = list(base_path.glob("**/info.json"))

    X = []
    y = []
    for file in info_files:
        data = json.loads(file.read_text())
        if not ("mean" in data and "v_phase" in data):
            continue
        vf = data["v_phase"]["11"]
        clr = data["chord_length_ratio"]
        X.append([vf, clr])
        y.append(data["homogenization"]["Thermal conductivity"]["value"])

    return np.array(X), np.array(y)


def accuracy_test(model, X_test, y_test):
    """
    Calculate the accuracy of the model on test data.

    Parameters:
    - model: Trained GPR model.
    - X_test (numpy.ndarray): Test data features.
    - y_test (numpy.ndarray): Test data target values.

    Returns:
    - float: Accuracy score.
    """
    # Predict mean for test data
    y_pred = model.predictmean(X_test)

    # Calculate whether each prediction is within the tolerance of the true value
    score = metrics.r2_score(y_true=y_test, y_pred=y_pred) * 100

    return score


gp_models = list(sorted(pathlib.Path("../adapt").glob("*.joblib"), key=lambda x: int(x.name.split("_")[2])))[:50][::2]

cm_subsection = np.linspace(0.1, 1, len(gp_models))
colors = [cm.binary(x) for x in cm_subsection]

accuracy = []
model_size = []


X_test, y_test = load_test_data(
    "/data/pirkelma/adaptive_gp_InCoKer/thermal_conductivity/20231215/validation_data/mean/"
    "test_data_32_thermal_conductivity"
)


for model_path, color in tqdm.tqdm(list(zip(gp_models, colors))):
    vf = np.arange(0.15, 0.9, 0.05)
    model = joblib.load(model_path)
    res = [model.predictmean(np.array([[v, 1.0]]))[0][0] for v in vf]

    plt.plot(vf, res, label=model_path.name, color=color)
    plt.xlabel("Volume fraction")
    plt.ylabel("Thermal conductivity")

    accuracy.append(accuracy_test(model, X_test, y_test))
    model_size.append(int(model_path.name.split("_")[2]))

plt.grid()
plt.legend()
plt.savefig("gp_evolution.png")

plt.figure()
plt.plot(model_size, accuracy)
plt.xlabel("#GP points")
plt.ylabel("Accuracy")
plt.savefig("accuracy_evolution.png")
