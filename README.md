# Inverse-Modeling-Ceramics-Thesis

This thesis aims to extend the findings of the paper [Top-down material design of multi-phase ceramics](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) by designing bottom up models for the design of ceramics. The repository consists of ongoing documentation of literature review and weekly progress, along with the code implementation.

The ongoing thesis report can be viewed [here](https://www.overleaf.com/project/65288161f5194816179d412b).


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
using the ``adapt_to_standard.py`` as:

```
python online_inverse_design/training/online_training.py adapt/model_name.joblib --export_model_file models/model_name.joblib
```

Using this model, inverse design can be performed using the scripts under the directory ``incoker-inv/online_inverse_design/optimization``. To 
perform validation designs, run the script:

```
python online_inverse_design/optimization/gradient_opt.py
```
