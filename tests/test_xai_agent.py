"""Unit tests for the XAI Refinement Agent utilities and callbacks."""

import json
import os
import tempfile
import pytest
from google.adk.agents import callback_context as callback_context_module
from unittest.mock import MagicMock

from machine_learning_engineering.shared_libraries import xai_util

def test_calculate_attribution_concentration():
    # Test normal inputs
    attributions = {"f1": 0.1, "f2": 0.8, "f3": 0.1}
    concentration = xai_util.calculate_attribution_concentration(attributions)
    assert pytest.approx(concentration) == 0.8 / 1.0
    
    # Test empty input
    assert xai_util.calculate_attribution_concentration({}) == 0.0
    
    # Test zero sum
    attributions_zero = {"f1": 0.0, "f2": 0.0}
    assert xai_util.calculate_attribution_concentration(attributions_zero) == 0.0
    
    # Test negative/absolute values
    attributions_neg = {"f1": -0.9, "f2": 0.1}
    assert pytest.approx(xai_util.calculate_attribution_concentration(attributions_neg)) == 0.9 / 1.0

def test_evaluate_leakage_risk_leaky():
    # Scenario: Top feature has 85% concentration and masking it increases RMSE from 10.0 to 30.0 (200% increase)
    metrics = {
        "feature_attributions": {"days_in_morgue": 0.85, "age": 0.15},
        "validation_score": 10.0,
        "masked_validation_score": 30.0,
        "lower_is_better": True
    }
    
    is_leaky, reason = xai_util.evaluate_leakage_risk(metrics, max_concentration_threshold=0.8, max_robustness_drop_pct=50.0)
    assert is_leaky is True
    assert "SUSPECTED TARGET LEAKAGE" in reason
    assert "days_in_morgue" in reason

def test_evaluate_leakage_risk_not_leaky_low_concentration():
    # Scenario: Distributed importances, even if masking a feature changes performance
    metrics = {
        "feature_attributions": {"f1": 0.33, "f2": 0.33, "f3": 0.34},
        "validation_score": 10.0,
        "masked_validation_score": 30.0,
        "lower_is_better": True
    }
    
    is_leaky, reason = xai_util.evaluate_leakage_risk(metrics, max_concentration_threshold=0.8, max_robustness_drop_pct=50.0)
    assert is_leaky is False
    assert "attribution is distributed reasonably" in reason

def test_evaluate_leakage_risk_not_leaky_robust():
    # Scenario: Top feature is highly dominant (90%) but masking it does not degrade performance (robust model)
    metrics = {
        "feature_attributions": {"f1": 0.90, "f2": 0.10},
        "validation_score": 1.0,
        "masked_validation_score": 1.05,
        "lower_is_better": True
    }
    
    is_leaky, reason = xai_util.evaluate_leakage_risk(metrics, max_concentration_threshold=0.8, max_robustness_drop_pct=50.0)
    assert is_leaky is False
    assert "Highly dominant but robustness drop did not exceed threshold" in reason

def test_parse_xai_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        # File doesn't exist
        assert xai_util.parse_xai_metrics(tmpdir) == {}

        # Valid JSON file
        metrics_data = {
            "feature_attributions": {"f1": 1.0, "f2": 0.2},
            "validation_score": 0.9,
            "masked_validation_score": 1.0,
            "lower_is_better": True,
        }
        with open(os.path.join(tmpdir, "xai_metrics.json"), "w") as f:
            json.dump(metrics_data, f)

        parsed = xai_util.parse_xai_metrics(tmpdir)
        assert parsed == metrics_data


def test_parse_xai_metrics_rejects_truncated_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "xai_metrics.json")
        with open(path, "w") as f:
            f.write('{\n  "feature_attributions": {\n    "longitude": ')
        assert xai_util.parse_xai_metrics(tmpdir) == {}


def test_to_json_float_numpy():
    import numpy as np

    assert xai_util.to_json_float(np.float32(1.5)) == pytest.approx(1.5)


def test_write_xai_metrics_atomic():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "xai_metrics.json")
        import numpy as np

        xai_util.write_xai_metrics(
            path,
            {
                "feature_attributions": {"f1": np.float32(0.8), "f2": np.float32(0.2)},
                "validation_score": np.float64(10.0),
                "masked_validation_score": np.float64(12.0),
                "lower_is_better": True,
            },
        )
        parsed = xai_util.parse_xai_metrics(tmpdir)
        assert parsed["feature_attributions"]["f1"] == pytest.approx(0.8)


def test_archive_xai_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_data = {
            "feature_attributions": {"f1": 0.8, "f2": 0.2},
            "validation_score": 10.0,
            "masked_validation_score": 12.0,
            "lower_is_better": True,
        }
        xai_util.write_xai_metrics(os.path.join(tmpdir, "xai_metrics.json"), metrics_data)

        archived = xai_util.archive_xai_metrics(tmpdir, "correction_audit_loop0")
        assert archived == os.path.join(
            tmpdir, "xai_metrics_archive", "correction_audit_loop0.json"
        )
        assert os.path.exists(archived)
        with open(archived, encoding="utf-8") as f:
            assert json.load(f)["feature_attributions"]["f1"] == pytest.approx(0.8)

        # Overwriting live metrics does not affect the archive copy.
        xai_util.write_xai_metrics(
            os.path.join(tmpdir, "xai_metrics.json"),
            {
                **metrics_data,
                "feature_attributions": {"f1": 0.1, "f2": 0.9},
            },
        )
        with open(archived, encoding="utf-8") as f:
            assert json.load(f)["feature_attributions"]["f1"] == pytest.approx(0.8)


def test_archive_xai_metrics_missing_source():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert xai_util.archive_xai_metrics(tmpdir, "missing") is None


def test_instrumentation_guide_formats_lower_is_better():
    # The shared instrumentation guide must embed the lower_is_better value so the
    # generated leakage-suite wiring scores in the right direction.
    from machine_learning_engineering.shared_libraries import xai_instrumentation_guide

    guide_true = xai_instrumentation_guide.build_instrumentation_instructions(
        lower_is_better=True
    )
    guide_false = xai_instrumentation_guide.build_instrumentation_instructions(
        lower_is_better=False
    )
    assert "True" in guide_true
    assert "False" in guide_false
    assert "__LOWER_IS_BETTER__" not in guide_true

def test_evaluate_leakage_risk_list_2d():
    # 2D list of attributions: 2 samples, 3 features
    metrics = {
        "feature_attributions": [
            [0.1, 0.8, 0.1],
            [0.3, 0.8, 0.1]
        ],
        "feature_names": ["f1", "f2", "f3"],
        "validation_score": 1.0,
        "masked_validation_score": 1.05,
        "lower_is_better": True,
    }
    is_leaky, reason = xai_util.evaluate_leakage_risk(metrics, max_concentration_threshold=0.7)
    # Mean attributions should be [0.2, 0.8, 0.1]
    # Concentration of f2 = 0.8 / 1.1 = 0.727 >= 0.7
    assert is_leaky is False
    assert "f2" in reason
    assert "0.73" in reason

def test_evaluate_leakage_risk_list_1d():
    # 1D list of attributions
    metrics = {
        "feature_attributions": [0.1, 0.8, 0.1],
        "feature_names": ["f1", "f2", "f3"],
        "validation_score": 1.0,
        "masked_validation_score": 1.05,
        "lower_is_better": True,
    }
    is_leaky, reason = xai_util.evaluate_leakage_risk(metrics, max_concentration_threshold=0.7)
    assert is_leaky is False
    assert "f2" in reason
