"""Tests for `incoker_inverse` package."""

import pathlib

import pytest

from incoker_inverse.online_inverse_design.optimization.gradient_opt import (
    main as opt_main,
)
from incoker_inverse.online_inverse_design.training.adapt_to_standard import (
    main as adapt_main,
)
from incoker_inverse.online_inverse_design.training.online_training import main


@pytest.fixture
def custom_config():
    """Return a custom configuration for testing."""
    return {
        "property_name": "young_modulus",
        "simulation_options": {"particle_quantity": 100, "dim": 16, "max_vertices": 10000},
        "parameterranges": {"VolumeFractionZirconia": [0.15, 0.85], "ChordLengthRatio": [0.3, 4.0]},
        "adaptive_phase_parameters": {
            "totalbudget": 1.0e20,
            "incrementalbudget": 1.0e5,
            "TOLFEM": 0.0,
            "TOLAcqui": 1.0,
            "TOLrelchange": 0,
            "TOL": 0,
        },
        "multiple_generation_options": {"usage": False, "num_generations": 1},
        "execpath": "./adapt",
        "validation_data_path": "./incoker_inverse/data/test_data_32_elasticity",
        "compute": True,
        "output_freq": 5,
        "initial_samples": 3,
        "max_samples": 5,
    }


@pytest.mark.online_training
def test_online_training_with_custom_config(custom_config):
    """Test the main function of online training with a custom configuration."""
    # Pass the custom configuration directly to the main function
    print("---Beginning adaptive GP test---")
    main(custom_config)


@pytest.mark.adapt_standard
def test_adapt_to_standard():
    """Test the main function of adaot to standard script."""
    print("---Beginning adaptive GP to sklearn conversion test---")
    adapt = pathlib.Path("adapt") / "final_gp_young_modulus.joblib"
    export = pathlib.Path("adapt") / "standard_gp.joblib"

    adapt_main(adapt, export, "young_modulus")


@pytest.mark.gradient_opt
def test_gradient_opt():
    """Test the main function of optimization script."""
    print("Beginning gradient optimization test")
    export = pathlib.Path("adapt") / "standard_gp.joblib"

    opt_main(property_name="young_modulus", property_value=250, model_file=export, multi_starts=5)
