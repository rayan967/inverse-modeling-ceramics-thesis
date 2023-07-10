import argparse
import pathlib
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


considered_features = [
    'volume_fraction_4', 'volume_fraction_10', 'volume_fraction_1',
    'chord_length_mean_4', 'chord_length_mean_10', 'chord_length_mean_1',
    'chord_length_variance_4', 'chord_length_variance_10', 'chord_length_variance_1'
]

# material properties to consider in training
considered_properties = [
    'thermal_conductivity',
    'young_modulus',
    'poisson_ratio',
    'thermal_expansion',
]


def main(train_data_file, export_model_file):

    training_data = pathlib.Path(train_data_file)
    if not training_data.exists():
        print(f"Error: training data path {training_data} does not exist.")

    if training_data.suffix == '.npy':
        data = np.load(training_data)
    else:
        print("Invalid data")

    print(f"loaded {data.shape[0]} training data pairs")

    data['thermal_expansion'] *= 1e6

    X, Y = extract_XY(data)

    assert Y.shape[0] == X.shape[0], "number of samples does not match"

    models = {}
    for i, property_name in enumerate(considered_properties):

        # pick a single property
        y = Y[:, i]

        # ignore NaNs in the data
        clean_indices = np.argwhere(~np.isnan(y))
        y_clean = y[clean_indices.flatten()]
        X_clean = X[clean_indices.flatten()]

        # create a pipeline object for training
        pipe = make_pipeline(
            StandardScaler(),   # scaler for data normalization
            GaussianProcessRegressor(kernel=Matern(), random_state=0)
        )

        # split in test and train data
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, random_state=0)
        pipe.fit(X_train, y_train)

        models[property_name] = {'pipe': pipe, 'features': considered_features}
        models[property_name]['X_train'] = X_train
        models[property_name]['X_test'] = X_test
        models[property_name]['y_train'] = y_train
        models[property_name]['y_test'] = y_test

        # Evaluate the model using the score method
        score = pipe.score(X_test, y_test)
        print("Model R-squared score on test set:", score)

        # Evaluate the model using the score method
        score = pipe.score(X_train, y_train)
        print("Model R-squared score on training set:", score)

        y_pred = pipe.predict(X_test)

        print(f"Property {property_name}")
        print("   score on train set = ", pipe.score(X_train, y_train))
        print("   score on test  set = ", pipe.score(X_test, y_test))

        # The mean squared error
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        models[property_name]['rmse'] = rmse
        print("   Root mean squared error: %.2f" % rmse)
        # The coefficient of determination: 1 is perfect prediction
        print("   Coefficient of determination: %.5f" % r2_score(y_test, y_pred))

        # cross validation
        cv = cross_validate(pipe, X_clean, y_clean, cv=5)
        cv_score_mean = cv['test_score'].mean()
        cv_score_std = cv['test_score'].std()
        models[property_name]['cv_score_mean'] = cv_score_mean
        models[property_name]['cv_score_std'] = cv_score_std
        print \
            ("   %0.5f accuracy with a standard deviation of %0.5f" % (cv_score_mean, cv_score_std))

    for prop in ['young_modulus', 'poisson_ratio', 'thermal_conductivity', 'thermal_expansion']:
        if prop in models:
            print \
                (f"{prop}: {models[prop]['cv_score_mean']:.3f}, {models[prop]['cv_score_std']:.1e}, {models[prop]['rmse']:.1e}")

    # export model for use in other projects
    if export_model_file is not None:
        from datetime import date
        import pkg_resources
        import sys
        installed_packages = pkg_resources.working_set
        installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
                                          for i in installed_packages])

        # store python code in current directory for reproducibility
        local_python_files = list(pathlib.Path().glob('*.py'))
        local_python_code = [f.read_text() for f in local_python_files]

        exported_model = {
            'models': models,
            'version_info':
                {
                    'date': date.today().isoformat(),
                    'python': sys.version,
                    'packages': installed_packages_list
                },
            'python_files': local_python_code
        }
        joblib.dump(exported_model, export_model_file)
    return models


def extract_XY(data):

    X = np.vstack(tuple(data[f] for f in considered_features)).T
    Y = np.vstack(tuple(data[p] for p in considered_properties)).T

    return X, Y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ML models using structure-property data of '
                                                 'RVEs.')
    parser.add_argument('train_data_file', type=pathlib.Path,
                        help='Path to the database of RVE structures and simulation results or '
                             'numpy file with training data already loaded')
    parser.add_argument('--export_model_file', type=pathlib.Path, required=False,
                        help='Path to a file where the trained models will be exported to.')
    args = parser.parse_args()

    main(args.train_data_file, args.export_model_file)


