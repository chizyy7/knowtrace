"""Tests for BKT model implementation."""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from knowtrace.models.bkt import BKT, save_bkt_params, load_bkt_params


def test_bkt_initialization():
    """Test BKT model initialization."""
    num_skills = 5
    bkt = BKT(num_skills=num_skills)

    # Check that parameters are initialized correctly
    assert bkt.num_skills == num_skills
    # We allocate num_skills + 1 for padding (index 0 reserved)
    assert len(bkt.p_l0) == num_skills + 1
    assert len(bkt.p_t) == num_skills + 1
    assert len(bkt.p_g) == num_skills + 1
    assert len(bkt.p_s) == num_skills + 1

    # Check default values (only for skill indices 1 through num_skills)
    assert np.allclose(bkt.p_l0[1:], 0.5)
    assert np.allclose(bkt.p_t[1:], 0.3)
    assert np.allclose(bkt.p_g[1:], 0.2)
    assert np.allclose(bkt.p_s[1:], 0.1)


def test_bkt_parameter_bounds():
    """Test that parameters stay within valid bounds."""
    num_skills = 3
    bkt = BKT(num_skills=num_skills)

    # Test setting parameters to extreme values (for valid skill indices only)
    # We need to provide arrays of length total_indices = num_skills + 1
    extreme_p_l0 = np.array([0.5, 0.0, 1.0, 0.5])  # index 0 unchanged, indices 1-3 modified
    extreme_p_t = np.array([0.3, 0.0, 1.0, 0.5])
    extreme_p_g = np.array([0.2, 0.0, 1.0, 0.5])
    extreme_p_s = np.array([0.1, 0.0, 1.0, 0.5])

    bkt.set_parameters(extreme_p_l0, extreme_p_t, extreme_p_g, extreme_p_s)

    # Check that parameters are clipped to valid range (only for skill indices 1 through num_skills)
    assert np.all(bkt.p_l0[1:] >= 0.001)
    assert np.all(bkt.p_l0[1:] <= 0.999)
    assert np.all(bkt.p_t[1:] >= 0.001)
    assert np.all(bkt.p_t[1:] <= 0.999)
    assert np.all(bkt.p_g[1:] >= 0.001)
    assert np.all(bkt.p_g[1:] <= 0.999)
    assert np.all(bkt.p_s[1:] >= 0.001)
    assert np.all(bkt.p_s[1:] <= 0.999)


def test_bkt_prediction_shape():
    """Test that BKT predictions have correct shape."""
    num_skills = 4
    bkt = BKT(num_skills=num_skills)

    # Test with single sequence (using skill indices 1 through num_skills)
    skill_seq = [1, 2, 3, 4]
    correct_seq = [1, 0, 1, 1]

    predictions = bkt.predict(skill_seq, correct_seq)
    assert len(predictions) == len(skill_seq)
    assert all(0 <= p <= 1 for p in predictions)


def test_bkt_prediction_consistency():
    """Test that BKT predictions are consistent."""
    num_skills = 3
    bkt = BKT(num_skills=num_skills)

    # Same sequence should give same predictions (using skill indices 1 through num_skills)
    skill_seq = [1, 2, 3, 1, 2]
    correct_seq = [1, 1, 0, 1, 0]

    pred1 = bkt.predict(skill_seq, correct_seq)
    pred2 = bkt.predict(skill_seq, correct_seq)

    assert np.allclose(pred1, pred2)


def test_bkt_skill_parameters():
    """Test getting skill-specific parameters."""
    num_skills = 3
    bkt = BKT(num_skills=num_skills)

    # Get parameters for each skill (using skill indices 1 through num_skills)
    for skill_id in range(1, num_skills + 1):
        p_l0, p_t, p_g, p_s = bkt._get_skill_parameters(skill_id)
        assert isinstance(p_l0, float)
        assert isinstance(p_t, float)
        assert isinstance(p_g, float)
        assert isinstance(p_s, float)
        assert 0 <= p_l0 <= 1
        assert 0 <= p_t <= 1
        assert 0 <= p_g <= 1
        assert 0 <= p_s <= 1


def test_bkt_parameter_save_load():
    """Test saving and loading BKT parameters."""
    num_skills = 3
    bkt = BKT(num_skills=num_skills)

    # Modify some parameters (for valid skill indices only)
    # Index 0 is reserved for padding and should remain unchanged
    bkt.p_l0[1] = 0.8  # skill 1
    bkt.p_t[2] = 0.4   # skill 2
    bkt.p_g[3] = 0.3   # skill 3

    # Get parameters
    original_params = bkt.get_parameters()

    # Test saving and loading
    with tempfile.TemporaryDirectory() as temp_dir:
        params_file = Path(temp_dir) / "bkt_params.json"
        save_bkt_params(original_params, params_file)

        loaded_params = load_bkt_params(params_file)

        # Check that loaded parameters match original
        for key in original_params.keys():
            np.testing.assert_allclose(
                original_params[key], loaded_params[key], rtol=1e-10
            )


def test_bkt_edge_cases():
    """Test BKT with edge cases."""
    num_skills = 2
    bkt = BKT(num_skills=num_skills)

    # Empty sequences
    predictions = bkt.predict([], [])
    assert predictions == []

    # Single observation (using skill index 1)
    predictions = bkt.predict([1], [1])
    assert len(predictions) == 1
    assert 0 <= predictions[0] <= 1

    # All correct (using skill indices 1 and 2)
    skill_seq = [1, 2, 1]
    correct_seq = [1, 1, 1]
    predictions = bkt.predict(skill_seq, correct_seq)
    assert len(predictions) == 3

    # All incorrect (using skill indices 1 and 2)
    skill_seq = [1, 2, 1]
    correct_seq = [0, 0, 0]
    predictions = bkt.predict(skill_seq, correct_seq)
    assert len(predictions) == 3


def test_bkt_knowledge_update():
    """Test the internal knowledge state update."""
    num_skills = 1
    bkt = BKT(num_skills=num_skills)

    # Test with known parameters
    p_l0, p_t, p_g, p_s = 0.5, 0.3, 0.2, 0.1

    # Start with unknown
    p_known_before = 0.0

    # After correct observation
    p_known_after = bkt._update_knowledge_state(
        p_known_before, 1, p_l0, p_t, p_g, p_s
    )
    # Should increase probability of knowing
    assert p_known_after > p_known_before

    # Start with known
    p_known_before = 1.0

    # After incorrect observation
    p_known_after = bkt._update_knowledge_state(
        p_known_before, 0, p_l0, p_t, p_g, p_s
    )
    # When certain of knowing and get incorrect, probability stays at 1.0
    # (due to slipping being the only explanation for incorrect when known)
    assert np.isclose(p_known_after, 1.0, atol=1e-10)