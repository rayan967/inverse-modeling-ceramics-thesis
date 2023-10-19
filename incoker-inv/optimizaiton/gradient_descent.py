import torch
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from pyDOE import lhs
import matplotlib.pyplot as plt

K_inv = None


def find_closest_point(Xt, point, selected_indices):
    distances = np.linalg.norm(Xt - point, axis=1)
    while True:
        index = np.argmin(distances)
        if index not in selected_indices:
            selected_indices.append(index)
            return Xt[index], selected_indices
        else:
            distances[index] = np.inf


def gradient_mean_wrapper(x, models, property_name):
    model = models[property_name]['pipe']
    features = models[property_name]['features']

    # Convert x to the relevant microstructure features
    microstructure_features = [x[i] for i in range(len(features))]
    X = np.array(microstructure_features).reshape(1, -1)

    # Access the scaler and GPR from the pipeline
    scaler = model.named_steps['standardscaler']
    gpr = model.named_steps['gaussianprocessregressor']

    # Scale the input data
    X_scaled = scaler.transform(X)

    # Compute the gradient using the scaled data and GPR
    return gpr_mean_grad(X_scaled, gpr)


def gradient_variance_wrapper(x, models, property_name):
    model = models[property_name]['pipe']
    features = models[property_name]['features']

    # Convert x to the relevant microstructure features
    microstructure_features = [x[i] for i in range(len(features))]
    X = np.array(microstructure_features).reshape(1, -1)

    # Access the scaler and GPR from the pipeline
    scaler = model.named_steps['standardscaler']
    gpr = model.named_steps['gaussianprocessregressor']

    # Scale the input data
    X_scaled = scaler.transform(X)

    # Compute the gradient using the scaled data and GPR
    return gpr_variance_grad(X_scaled, gpr)


def convert_x_to_microstructure(x, features):
    if len(features) == 9:
        volume_fraction_4 = x[0]
        volume_fraction_10 = x[1]
        volume_fraction_1 = x[2]
        chord_length_mean_4 = x[3]
        chord_length_mean_10 = x[4]
        chord_length_mean_1 = x[5]
        chord_length_variance_4 = x[6]
        chord_length_variance_10 = x[7]
        chord_length_variance_1 = x[8]

        # Construct the microstructure representation
        microstructure = {
            'volume_fraction_4': volume_fraction_4,
            'volume_fraction_10': volume_fraction_10,
            'volume_fraction_1': volume_fraction_1,
            'chord_length_mean_4': chord_length_mean_4,
            'chord_length_mean_10': chord_length_mean_10,
            'chord_length_mean_1': chord_length_mean_1,
            'chord_length_variance_4': chord_length_variance_4,
            'chord_length_variance_10': chord_length_variance_10,
            'chord_length_variance_1': chord_length_variance_1,
        }
    elif len(features) == 8:
        volume_fraction_4 = x[0]
        volume_fraction_1 = x[1]
        chord_length_mean_4 = x[2]
        chord_length_mean_10 = x[3]
        chord_length_mean_1 = x[4]
        chord_length_variance_4 = x[5]
        chord_length_variance_10 = x[6]
        chord_length_variance_1 = x[7]

        # Construct the microstructure representation
        microstructure = {
            'volume_fraction_4': volume_fraction_4,
            'volume_fraction_1': volume_fraction_1,
            'chord_length_mean_4': chord_length_mean_4,
            'chord_length_mean_10': chord_length_mean_10,
            'chord_length_mean_1': chord_length_mean_1,
            'chord_length_variance_4': chord_length_variance_4,
            'chord_length_variance_10': chord_length_variance_10,
            'chord_length_variance_1': chord_length_variance_1,
        }
    elif len(features) == 3:
        volume_fraction_4 = x[0]
        volume_fraction_1 = x[1]
        chord_length_ratio = x[2]

        # Construct the microstructure representation
        microstructure = {
            'volume_fraction_4': volume_fraction_4,
            'volume_fraction_1': volume_fraction_1,
            'chord_length_ratio': chord_length_ratio,
        }
    elif len(features) == 2:
        volume_fraction_4 = x[0]
        chord_length_ratio = x[1]

        # Construct the microstructure representation
        microstructure = {
            'volume_fraction_4': volume_fraction_4,
            'chord_length_ratio': chord_length_ratio,
        }

    return microstructure

class OptimizationResult:
    def __init__(self, x, fun, jac, success, message, nfev, nit):
        self.x = x
        self.fun = fun
        self.jac = jac
        self.success = success
        self.message = message
        self.nfev = nfev
        self.nit = nit

def torch_minimize(fun, jac, x0, bounds, max_iter=1000, tol=1e-6):
    x = torch.tensor(x0, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([x])

    for _ in range(max_iter):
        optimizer.zero_grad()
        loss = fun(x)

        gradient = jac(x)
        x.grad = torch.tensor(gradient, dtype=torch.float32)

        optimizer.step()

        # Projection step to ensure bounds
        with torch.no_grad():
            for i, (min_val, max_val) in enumerate(bounds):
                x[i].clamp_(min_val, max_val)

        # Stopping criterion based on gradient norm
        grad_norm = torch.norm(x.grad)
        if grad_norm < tol:
            break

    # Return in a format similar to scipy.optimize.minimize
    result = OptimizationResult(
        x=x.detach().numpy(),
        fun=loss.item(),
        jac=gradient,
        success=grad_norm < tol,
        message="Converged" if grad_norm < tol else "Max iterations reached",
        nfev=_ + 1,
        nit=_ + 1
    )

    return result


def objective_function(x, desired_property, models, property_name, callback=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()

    if callback is not None:
        callback(x)
    features = models[property_name]['features']
    microstructure = convert_x_to_microstructure(x, features)
    predicted_property, uncertainty = predict_property(property_name, microstructure, models, uncertainty=True)

    discrepancy = predicted_property - desired_property

    return (discrepancy ** 2) + uncertainty[0]*1e-15


def objective_gradient(x, desired_property, models, property_name):
    features = models[property_name]['features']
    gpr_grad = gradient_mean_wrapper(x.detach().numpy(), models, property_name)
    gpr_var_grad = gradient_variance_wrapper(x.detach().numpy(), models, property_name)
    predicted_property = predict_property(property_name, convert_x_to_microstructure(x.detach().numpy(), features), models)
    discrepancy = predicted_property - desired_property

    return (2 * discrepancy * gpr_grad) + gpr_var_grad*1e-15


def gpr_mean_grad(X_test, gpr):
    X_train = gpr.X_train_
    kernel = gpr.kernel_

    kernel_1, white = kernel.k1, kernel.k2
    alpha = gpr.alpha_
    gradients = []
    for x_star in X_test:
        # Compute the gradient for x_star across all training data
        # Only need grad of kernel_1, white is constant
        k_gradient_matrix = kernel_1.gradient_x(x_star, X_train)

        # Multiply the gradient matrix with alpha and sum across training data
        grad_sum = np.dot(alpha, k_gradient_matrix)

        # Adjust for normalization
        grad_sum_adjusted = gpr._y_train_std * grad_sum

        gradients.append(grad_sum_adjusted)

    return np.array(gradients).ravel()

def gpr_variance_grad(X_test, gpr):
    global K_inv
    X_train = gpr.X_train_
    kernel = gpr.kernel_

    # Decompose the kernel into its constituent parts
    kernel_1, white = kernel.k1, kernel.k2

    if K_inv is None:
        # L.L^T = K (with noise)
        # K_inv = L^-T.L^-1
        L_inv = np.linalg.inv(gpr.L_)
        K_inv = L_inv.T.dot(L_inv)

    gradients = []
    for x_star in X_test:
        dk_xx = kernel_1.gradient_x(x_star, x_star.reshape(1, -1))

        # Compute the kernel vector k(x_star)
        k_x_star = kernel_1(x_star.reshape(1, -1), X_train).ravel()
        # Compute the gradient of kernel vector
        dk_x_star = kernel_1.gradient_x(x_star, X_train)
        # Compute the gradient of variance
        grad_variance = dk_xx - 2 * np.dot(np.dot(k_x_star.T, K_inv), dk_x_star)
        grad_variance_adjusted = grad_variance * gpr._y_train_std**2

        gradients.append(grad_variance_adjusted)

    return np.array(gradients).ravel()


def predict_property(property_name, microstructure, models, uncertainty=False):
    model = models[property_name]['pipe']
    features = models[property_name]['features']


    microstructure_features = [microstructure[feature] for feature in features]

    X = np.array(microstructure_features).reshape(1, -1)

    # Use the GPR model to predict the property value
    predicted_value, std_dev = model.predict(X, return_std=True)

    if uncertainty:
        return predicted_value[0], std_dev
    else:
        return predicted_value[0]



def optimise_for_value(prop, X, property_name):
    # Random sample starting points
    #initial_points = X[np.random.choice(X.shape[0], NUM_STARTS, replace=False)]
    # LHS sampling for25 uniform starting points in multi-start optimization
    num_samples = 30
    lhs_samples = lhs(len(bounds), samples=num_samples)
    for i in range(len(bounds)):
        lhs_samples[:, i] = lhs_samples[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    # Find the closest points in X to the LHS samples
    selected_indices = []
    Xt_initial = np.zeros((num_samples, X.shape[1]))  # Initialize closest points array
    for i in range(num_samples):
        Xt_initial[i], selected_indices = find_closest_point(X, lhs_samples[i], selected_indices)
    initial_points = X[selected_indices]

    #indices = np.where(np.isclose(X[:, 0], 0.0070801))
    #initial_points = X[selected_indices]
    print(bounds)
    best_result = None
    best_value = float('inf')
    best_iterates = []
    best_f_values = []


    for initial_point in initial_points:
        current_iterates = []
        current_f_values = []

        def callback(x):
            current_iterates.append(np.copy(x))
            f_val = objective_function(x, prop, models, property_name, None)
            current_f_values.append(f_val)

        res = torch_minimize(
            fun=lambda x: objective_function(x, prop, models, property_name, callback),
            jac=lambda x: objective_gradient(x, prop, models, property_name),
            x0=initial_point,
            bounds=bounds,
        )
        if res.fun < best_value:
            best_value = res.fun
            best_result = res
            best_iterates = current_iterates.copy()
            best_f_values = current_f_values.copy()


    optimal_x = best_result.x
    optimal_microstructure = convert_x_to_microstructure(optimal_x, features)
    optimal_property_value = predict_property(property_name, optimal_microstructure, models)
    print(optimal_microstructure)
    print("Error in optimisation: " + str(np.abs(prop - optimal_property_value)))

    print("Iter\tX1\t\t\tX2\t\t\tf(X)")

    for i, (iterate, f_val) in enumerate(zip(best_iterates, best_f_values)):
        print(f"{i+1}\t{iterate[0]:.6f}\t{iterate[1]:.6f}\t{f_val:.6f}")


    if len(features) <= 2:
        X = models[property_name]['X_train']
        # Dense grid
        v2_values = np.linspace(min(X[:, 0]), max(X[:, 0]), num=100)
        rho_values = np.linspace(min(X[:, 1]), max(X[:, 1]), num=100)
        v2_grid, rho_grid = np.meshgrid(v2_values, rho_values)

        feature_grid = np.vstack([v2_grid.ravel(), rho_grid.ravel()]).T

        predictions = []
        for feature in feature_grid:
            value = objective_function(feature, prop, models, property_name)
            predictions.append(value)

        predictions = np.array(predictions)

        predictions_grid = predictions.reshape(v2_grid.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(v2_grid, rho_grid, predictions_grid, alpha=0.5)  # Reduced alpha to see points clearly

        # Labels and title
        ax.set_xlabel('Volume Fraction (zirconia)')
        ax.set_ylabel('Particle Size Ratio')
        ax.set_zlabel(str(property_name) + " = " + str(prop))
        ax.set_title('3D Surface of Objective function')

        # Plotting the iterates
        iterates = np.array(best_iterates)
        num_iterates = len(iterates)
        colors = plt.cm.coolwarm(np.linspace(0, 1, num_iterates))

        for i in range(num_iterates):
            x_val, y_val = iterates[i, 0], iterates[i, 1]
            z_val = (models[property_name]['pipe'].predict([[x_val, y_val]]) - prop) ** 2
            ax.scatter(x_val, y_val, z_val, c=[colors[i]], s=20)

        # Show the plot
        plt.show()


print("starting opt")

property_name = 'thermal_conductivity'

# Change next line for different feature sets from models folder
models = joblib.load("models/2d_model.joblib")["models"]

X = models[property_name]['X_train']
X_test = models[property_name]['X_test']
Y = models[property_name]['y_train']
y_test = models[property_name]['y_test']

# Compute the minimum and maximum values for each feature in the training data
min_values = np.min(X, axis=0)
max_values = np.max(X, axis=0)
features = models[property_name]['features']
print("Features: ", str(features))

# Define the search space dimensions based on the minimum and maximum values
bounds = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

cons = [{'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1},
        {'type': 'ineq', 'fun': lambda x: -x[2] + 0.01},]

dimensions = [(min_val, max_val) for min_val, max_val in zip(min_values, max_values)]

NUM_STARTS = 30  # Number of starting points for multi-start

prop_bounds = (min(Y), max(Y))
# LHS sampling for uniform starting points in multi-start optimization
num_samples = 30
lhs_samples = lhs(len(bounds), samples=num_samples)
for i in range(len(bounds)):
    lhs_samples[:, i] = lhs_samples[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

# Find the closest points in X to the LHS samples
selected_indices = []
Xt_initial = np.zeros((num_samples, X.shape[1]))  # Initialize closest points array
for i in range(num_samples):
    Xt_initial[i], selected_indices = find_closest_point(X, lhs_samples[i], selected_indices)
initial_points = X[selected_indices]


# Try optimising one value
optimise_for_value(12, X, property_name)
#########
# or random choice
# initial_points = X[np.random.choice(X.shape[0], NUM_STARTS, replace=False)]


## Now optimizing subspace
# Grid of 100 points across property bounds for plot
num_points = 100
prop_values = np.linspace(prop_bounds[0], prop_bounds[1], num_points)


optimal_microstructures = []
optimal_volume_fractions_4 = []
optimal_properties = []
optimal_rho = []

actual_volume_fractions_4 = X[:, 0]
actual_rho = X[:, 1]
actual_properties = Y
count = 0
error_bars_min = []
error_bars_max = []

for prop in prop_values:
    count += 1
    best_result = None
    best_value = float('inf')

    for initial_point in initial_points:
        res = torch_minimize(
            fun=lambda x: objective_function(x, prop, models, property_name),
            jac=lambda x: objective_gradient(x, prop, models, property_name),
            x0=initial_point,
            bounds=bounds,
        )
        if res.fun < best_value:
            best_value = res.fun
            best_result = res

    optimal_x = best_result.x
    optimal_microstructure = convert_x_to_microstructure(optimal_x, features)

    optimal_microstructures.append(optimal_microstructure)
    # Store the optimal volume fraction and thermal expansion value
    optimal_volume_fractions_4.append(optimal_microstructure['volume_fraction_4'])
    optimal_rho.append(optimal_microstructure['chord_length_ratio'])
    optimal_properties.append(prop)


predicted_optimal_properties = []
predicted_actual_properties = []

predicted_properties = []

for optimal_microstructure in optimal_microstructures:
    predicted_value = predict_property(property_name, optimal_microstructure, models)
    predicted_properties.append(predicted_value)

predicted_properties = np.array(predicted_properties)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(prop_values, predicted_properties))
print(f"Root Mean Square Error (RMSE) between predicted and desired values: {rmse}")

# Calculate R2 score for optimized and actual feature sets
r2 = r2_score(prop_values, predicted_properties)
print(f"R2 score between predicted and desired values: {r2*100}")


plt.figure()
plt.scatter(actual_volume_fractions_4, actual_properties, label="Actual",  color='blue', marker='o')
plt.scatter(optimal_volume_fractions_4, optimal_properties, label="Optimized", color='red', marker='x')
plt.xlabel("Volume Fraction Zirconia")
plt.ylabel(property_name)
plt.legend()
plt.show()
##################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for actual points
ax.scatter(actual_volume_fractions_4, actual_rho, actual_properties, label="Actual", color='blue', marker='o')

# Scatter plot for optimized points
ax.scatter(optimal_volume_fractions_4, optimal_rho, optimal_properties, label="Optimized", color='red', marker='x')

# Set labels
ax.set_xlabel('Volume Fraction Zirconia')
ax.set_ylabel('Particle Size Ratio (rho)')
ax.set_zlabel(property_name)

# Show legend
ax.legend()

plt.show()
###############
