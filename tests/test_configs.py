#!/usr/bin/env python3
"""
Verify that every ExperimentConfigs preset matches its expected feature schema.

Runs both as a pytest module (`pytest tests/test_configs.py`) and as a
standalone script (`python tests/test_configs.py`).
"""

import os
import sys

import pytest

# Add hippie to path
code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hippie'))
sys.path.append(code_dir)

from multimodal_model import ExperimentConfigs

# Expected feature values for each named ExperimentConfigs preset. Keys must be
# real ExperimentConfigs staticmethods (see hippie/multimodal_model.py).
EXPECTED_CONFIGS = {
    "baseline": {
        "use_source_embedding": False,
        "use_class_embedding": False,
        "use_fusion_encoder": False,
        "use_batch_norm": False,
        "use_augmentations": False,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "with_source": {
        "use_source_embedding": True,
        "use_class_embedding": False,
        "use_fusion_encoder": True,
        "use_batch_norm": False,
        "use_augmentations": False,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "with_class": {
        "use_source_embedding": False,
        "use_class_embedding": True,
        "use_fusion_encoder": True,
        "use_batch_norm": False,
        "use_augmentations": False,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "with_both_embeddings": {
        "use_source_embedding": True,
        "use_class_embedding": True,
        "use_fusion_encoder": True,
        "use_batch_norm": False,
        "use_augmentations": False,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "with_light_augmentations": {
        "use_source_embedding": False,
        "use_class_embedding": False,
        "use_fusion_encoder": False,
        "use_batch_norm": False,
        "use_augmentations": True,
        "augment_prob": 0.3,
        "noise_std": 0.03,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "with_heavy_augmentations": {
        "use_source_embedding": True,
        "use_class_embedding": True,
        "use_fusion_encoder": False,
        "use_batch_norm": False,
        "use_augmentations": True,
        "augment_prob": 0.7,
        "noise_std": 0.08,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "with_batch_norm": {
        "use_source_embedding": True,
        "use_class_embedding": True,
        "use_fusion_encoder": True,
        "use_batch_norm": True,
        "use_augmentations": True,
        "augment_prob": 0.3,
        "noise_std": 0.03,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "no_fusion": {
        "use_source_embedding": True,
        "use_class_embedding": True,
        "use_fusion_encoder": False,
        "use_batch_norm": False,
        "use_augmentations": False,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "no_augmentations": {
        "use_source_embedding": True,
        "use_class_embedding": True,
        "use_fusion_encoder": True,
        "use_batch_norm": True,
        "use_augmentations": False,
        "class_embedding_dropout": 0.0,
        "reconstruction_consistency_weight": 0.0,
    },
    "full_architecture": {
        "use_source_embedding": True,
        "use_class_embedding": True,
        "use_fusion_encoder": True,
        "use_batch_norm": True,
        "use_augmentations": True,
        "augment_prob": 0.3,
        "noise_std": 0.03,
        "class_embedding_dropout": 0.3,
        "reconstruction_consistency_weight": 0.15,
        "embedding_warmup_epochs": 5,
    },
}


def check_config(name, expected_features):
    """Return (ok, errors) for a single named config against expected features."""
    config = getattr(ExperimentConfigs, name)()
    errors = []
    for feature, expected_value in expected_features.items():
        actual_value = getattr(config, feature)
        if actual_value != expected_value:
            errors.append(f"{feature}: got {actual_value}, expected {expected_value}")
    return (len(errors) == 0), errors


@pytest.mark.parametrize("name,expected", sorted(EXPECTED_CONFIGS.items()))
def test_config_matches_schema(name, expected):
    """Each ExperimentConfigs preset must match its documented feature schema."""
    ok, errors = check_config(name, expected)
    assert ok, f"{name}: " + "; ".join(errors)


def main():
    """Standalone runner that prints a per-config pass/fail summary."""
    print("=" * 60)
    print("Testing All Configuration Schemas")
    print("=" * 60)

    results = {}
    for config_name, expected in EXPECTED_CONFIGS.items():
        ok, errors = check_config(config_name, expected)
        results[config_name] = ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {config_name}")
        for error in errors:
            print(f"    - {error}")

    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
