# Inverse-Modeling-Ceramics-Thesis

This thesis aims to extend the findings of the paper [Top-down material design of multi-phase ceramics](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) by designing bottom up models for the design of ceramics. The repository consists of ongoing documentation of literature review and weekly progress, along with the code implementation.

The completed thesis can be viewed [here](https://drive.google.com/file/d/1hwqsXUgSolrxIaKnrLP7zDHVB9zoaZii/view?usp=sharing).


### Installation
The app and all required dependencies can be cloned from the Gitlab Package registry:
```
git clone https://oauth2:<your_personal_token>@gitlab.cc-asp.fraunhofer.de/ray29582/inverse-modeling-ceramics-thesis.git
```

Once you have a copy of the source, you can install it in a ``poetry`` environment using:
```
pip install poetry
poetry install
```
This will install the project package and its dependencies.

You can then move directory to the internal project files:
```
cd incoker-inv
```


### Online Training and Optimization

The files for an online training run are located at the directory ``incoker-inv/online_inverse_design/training``. To start an
online adaptive training, run the script:

```
python online_inverse_design/training/online_training.py
```

This will start an interative GPR training for the desired material property. Once training is complete (typically 12 hours), 
the model files will be stored at the directory ``incoker-inv/adapt``. The saved GPR can then be refactored into the skopt GPR 
using the ``adapt_to_standard.py`` script as:

```
python online_inverse_design/training/adapt_to_standard.py adapt/model_name.joblib --export_model_file models/model_name.joblib
```

Using this model, inverse design can be performed using the scripts under the directory ``incoker-inv/online_inverse_design/optimization``. To 
perform validation designs, run the script:

```
python online_inverse_design/optimization/gradient_opt.py --model_file <path_to_surrogate_model.joblib> --property_name <property_name> --property_value <desired_value>
```


### Offline Inverse Design

Using pregenerated offline data, many of the initial tests were performed. The relevant files are located at the directory ``incoker-inv/offline_inverse_design``.

To run the grid search for finding optimal hyperparameters for each property run the script:

```
python offline_inverse_design/training/grid_search.py
```
A grid to evaluate the performance of forward training across different feature sets can be performed using the script:

```
python offline_inverse_design/training/feature_set_viz.py
```

Based on the results, training with the optimized parameters (manually changed in the script) can be done using the script:

```
python offline_inverse_design/training/standard_training.py <path_to_training_data.npy> --export_model_file <path_to_export_models.joblib> --number_of_features <number_of_features (2, 3, or 8)>
```

Once the models are trained and stored, evaluations of the inverse design using the model can be performed using the script:
```
python offline_inverse_training/optimization/gradient_opt.py --model_file <path_to_surrogate_model.joblib> --property_name <property_name> --property_value <desired_value>
```

A full comparison of the different kernels (trained independently) for inverse design can be performed using the script:
```
python offline_inverse_training/optimization/kernel_comparison.py
```

### Direct Inverse Training

The DIT described in the thesis can be performed and evaluated using the files located at the directory ``incoker-inv/direct_inverse_training``.
To run the training script (``incoker-inv/direct_inverse_training/reverse_training.py``), you need a dataset file (in .npy format) containing the RVE structure-property
data. You can specify an export file path to save the trained models. Use the following
command to execute the script:

```
python direct_inverse_training/reverse_training.py <path_to_training_data.npy> --export_model_file <path_to_export_models.joblib>
```