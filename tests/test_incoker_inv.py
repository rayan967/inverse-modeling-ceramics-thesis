"""Tests for `incoker_inverse` package."""

import pytest
from incoker_inverse.online_inverse_design.training.online_training import main


@pytest.fixture
def custom_config():
    """Return a custom configuration for testing."""
    return {
        "property_name": "thermal_conductivity",
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
        "validation_data_path": "/data/pirkelma/adaptive_gp_InCoKer/thermal_conductivity/20231215/validation_data/mean"
        "/test_data_32_thermal_conductivity",
        "compute": True,
        "output_freq": 5,
        "initial_samples": 3,
        "max_samples": 4,
    }


@pytest.mark.online_training
def test_online_training_with_custom_config(custom_config):
    """Test the main function of online training with a custom configuration."""
    # Pass the custom configuration directly to the main function
    main(custom_config)
