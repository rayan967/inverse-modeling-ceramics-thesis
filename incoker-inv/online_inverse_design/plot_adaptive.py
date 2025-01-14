import os
import sys
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
from training.online_training import *


def predict_property(microstructure):
    model = joblib.load(gp_file)
    features = ["volume_fraction_4", "chord_length_ratio"]
    microstructure_features = [microstructure[feature] for feature in features]
    X = np.array(microstructure_features).reshape(1, -1)
    # Use the GPR model to predict the property value
    predicted_value = model.predictmean(X)
    print(predicted_value[0])
    return predicted_value[0]


def convert_x_to_microstructure(x):
    volume_fraction_4 = x[0]
    chord_length_ratio = x[1]

    # Construct the microstructure representation
    microstructure = {
        "volume_fraction_4": volume_fraction_4,
        "chord_length_ratio": chord_length_ratio,
    }
    print(microstructure)
    return microstructure


def main():
    gp = joblib.load(gp_file)
    features = ["volume_fraction_4", "chord_length_ratio"]

    v2_values = np.linspace(0.1, 0.9, num=100)
    rho_values = np.linspace(0.3, 4.0, num=100)
    v2_grid, rho_grid = np.meshgrid(v2_values, rho_values)

    # Flatten the grid to pass it to the model for prediction
    feature_grid = np.vstack([v2_grid.ravel(), rho_grid.ravel()]).T

    # Assume 'model' is your trained 2-feature model
    # Make predictions
    predictions = gp.predictmean(feature_grid)

    # Reshape the predictions to match the shape of the grid
    predictions_grid = predictions.reshape(v2_grid.shape)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(v2_grid, rho_grid, predictions_grid)

    # Add labels and title
    ax.set_xlabel("Volume Fraction (zirconia)")
    ax.set_ylabel("Particle Size Ratio")
    ax.set_zlabel("Thermal conductivity")
    ax.set_title("3D Surface of Predicted Material Property")

    plt.show()

    zr_values = np.linspace(0.1, 0.9)

    chord_length_values = np.full(len(zr_values), 1)
    features = np.stack(
        (zr_values, chord_length_values),
        axis=1,
    )

    predictions = gp.predictmean(features)

    plt.scatter(zr_values, predictions, label="Actual", color="blue", marker="o")

    plt.xlabel("Volume Fraction Zirconia")
    plt.ylabel("Thermal conductivity")
    plt.legend()
    plt.show()


import matplotlib.pyplot as plt
import joblib


def plot_design_space():
    # Load the Gaussian Process model
    gp = joblib.load(gp_file)

    # Extract the training data points
    X = gp.X

    # Design space limits
    x_min, x_max = 0.1, 0.9  # First dimension limits
    y_min, y_max = 0.3, 4.0  # Second dimension limits

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], color="blue", marker="o", label="Training Points")

    # Setting plot limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Adding labels and title
    plt.xlabel("First Dimension (e.g., Volume Fraction)")
    plt.ylabel("Second Dimension (e.g., Particle Size Ratio)")
    plt.title("Training Data Points in Design Space")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_training_data():
    # Load the Gaussian Process model
    gp = joblib.load(gp_file)

    # Extract the training data points
    X = gp.X
    y_pred = gp.predictmean(X)

    # Design space limits
    x_min, x_max = 0.1, 0.9  # First dimension limits
    y_min, y_max = 0.3, 5.0  # Second dimension limits

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], gp.yt[:, 0], color="blue", marker="o", label="Training Points")
    plt.scatter(X[:, 0], y_pred, label="Predicted", color="red", marker="x")

    print("Point\tX1\t\t\tX2\t\t\tf(X)")
    for i, (point, f_val) in enumerate(zip(gp.X, gp.yt)):
        f_val_scalar = f_val[0] if isinstance(f_val, (list, np.ndarray)) and len(f_val) == 1 else f_val
        print(f"{i + 1}\t{point[0]:.6f}\t{point[1]:.6f}\t{f_val_scalar:.6f}")

    # Adding labels and title
    plt.xlabel("First Dimension (e.g., Volume Fraction)")
    plt.ylabel("Second Dimension (e.g., Property)")
    plt.title("Training Data Points in Design Space")

    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    gp_file = "adapt/7_gp.joblib"
    # parser = argparse.ArgumentParser(description='Predict property')
    # parser.add_argument('prop1', type=float,
    # help='Expected property in JSON format')
    # parser.add_argument('prop2', type=float,
    # help='Expected property in JSON format')
    # args = parser.parse_args()
    # arr = np.array([args.prop1,args.prop2])
    plot_training_data()
    plot_design_space()
    # microstr = convert_x_to_microstructure(arr)
    # predict_property(microstr)
    main()
