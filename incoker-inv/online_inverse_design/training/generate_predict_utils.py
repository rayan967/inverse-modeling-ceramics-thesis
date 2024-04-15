"""Generate and predict material properties using incoker-micro-sims package."""

import sys
from pathlib import Path

from sklearn import metrics

current_file = Path(__file__).resolve()
run_directory = current_file.parent.parent.parent
sys.path.append(str(run_directory))
import numpy as np
from incoker_micro_sims import prediction_pipeline

phase_zirconia = 11
phase_alumina = 4
# material properties to consider in training
considered_properties = [
    "thermal_conductivity",
    "thermal_expansion",
    "young_modulus",
    "poisson_ratio",
]

# For result indexing
property_dict = {
    "thermal_conductivity": "Thermal conductivity",
    "thermal_expansion": "Thermal expansion",
    "young_modulus": "Young modulus",
    "poisson_ratio": "Poisson ratio",
}

# For simulation options
property_dict_category = {
    "thermal_conductivity": "thermal_conductivity",
    "thermal_expansion": "thermal_expansion",
    "young_modulus": "elasticity",
    "poisson_ratio": "elasticity",
}


def generate_candidate_point(
    input, simulation_options, property_name, output_stream, runpath, run_phase, mul_generate_options, parameterranges
):
    """Generate and predict material property of a candidate structure."""
    if mul_generate_options["usage"]:
        num_generations = mul_generate_options["num_generations"]
    else:
        num_generations = 1

    output_stream.error_detected = False  # Set error flag
    output_path = Path(runpath, run_phase, f"v={input[0]:.2f},r={input[1]:.2f}")
    output_path.mkdir(parents=True, exist_ok=True)
    simulation_options["output_path"] = output_path
    simulation_options["n_rves"] = num_generations
    simulation_options["n_workers"] = max(min(num_generations, 16), 1)

    all_results = prediction_pipeline.generate_and_predict(input, simulation_options)

    vfs = []
    clrs = []
    values = []
    for rve_path, result in all_results.items():
        output_value = get_output(result, property_name)

        vf = result["v_phase"][str(phase_zirconia)]
        cl_11 = result["chord_length_analysis"]["phase_chord_lengths"][str(phase_zirconia)]["mean_chord_length"]
        cl_4 = result["chord_length_analysis"]["phase_chord_lengths"][str(phase_alumina)]["mean_chord_length"]
        clr = cl_11 / cl_4

        vfs.append(vf)
        clrs.append(clr)
        values.append(output_value)

        print(f"vf:\t {vf:.2f}, clr:\t {clr:.2f}, v:\t {output_value:.2f}")

    vfs = np.array(vfs)
    clrs = np.array(clrs)
    values = np.array(values)

    print(f"volume fraction    : {vfs.mean():.2f} +- {vfs.var():.2f}")
    print(f"chord length ratio : {clrs.mean():.2f} +- {clrs.var():.2f}")
    print(f"material properties: {values.mean():.2f} +- {values.var():.2f}")

    # Check for Mapdl error
    if output_stream.error_detected:
        # Reset error flag
        output_stream.error_detected = False
        raise Exception("Error detected during operation: Mapdl")

    generated_points = np.array([vfs, clrs]).T
    yt_samples = values
    weights = calculate_weights(parameterranges)

    # Calculate variance and mean of outputs if we have enough samples
    if len(yt_samples) >= 1:
        # Calculate variance and mean of outputs if we have enough samples
        if mul_generate_options["usage"]:
            if len(yt_samples) == 1:
                variance = 1e-1
            else:
                variance = np.var(yt_samples, ddof=1)  # Using sample variance
        else:
            variance = 1e-4

        # Calculate weighted distances and select the best point
        distances = [weighted_distance(input, Xg, weights) for Xg in generated_points]
        best_index = np.argmin(distances)
        best_X = generated_points[best_index]
        best_y = yt_samples[best_index]

    else:
        raise Exception("Not enough samples were generated.")

    return best_X, best_y, variance


def get_output(result, property_name):
    """Get output based on property name."""
    # For CTE
    if property_name == "thermal_expansion":
        output_value = result["mean"]

    # For the rest
    else:
        output_value = result["homogenization"][property_dict[property_name]]["value"]
    return output_value


def accuracy_test(model, X_test, y_test, tolerance=1e-2):
    """
    Calculate the accuracy of the model on test data.

    Parameters:
    - model: Trained GPR model.
    - X_test (numpy.ndarray): Test data features.
    - y_test (numpy.ndarray): Test data target values.
    - tolerance (float, optional): Tolerance for accuracy. Defaults to 1E-2.

    Returns:
    - float: Accuracy score.
    """
    # Predict mean for test data
    y_pred = model.predictmean(X_test)

    # Calculate whether each prediction is within the tolerance of the true value
    score = metrics.r2_score(y_true=y_test, y_pred=y_pred) * 100

    return score


def calculate_weights(parameterranges):
    """
    Calculate weights for each parameter inversely proportional to their range.

    :param parameterranges: Array of parameter ranges.
    :return: Array of weights.
    """
    ranges = parameterranges[:, 1] - parameterranges[:, 0]
    weights = 1 / ranges
    return weights


def weighted_distance(point_a, point_b, weights):
    """
    Calculate the weighted Euclidean distance between two points.

    :param point_a: First point (array-like).
    :param point_b: Second point (array-like).
    :param weights: Weights for each dimension (array-like).
    :return: Weighted distance.
    """
    diff = np.array(point_a) - np.array(point_b)
    weighted_diff = diff * weights
    return np.sqrt(np.sum(weighted_diff**2))
