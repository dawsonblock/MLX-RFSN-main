#!/usr/bin/env python3
"""
RFSN v10 — Layer policy acceptance tests.

Verifies that LayerPolicy falls back to defaults, validates inputs,
produces deterministic results, and loads correctly from JSON.
"""
from __future__ import annotations

import json

import pytest

from rfsn_v10.quantization.layer_policy import (
    KNOWN_MODES,
    LayerPolicy,
    load_policy,
    validate_layer_policy,
)

# ------------------------------------------------------------------
# LayerPolicy tests
# ------------------------------------------------------------------

class TestLayerPolicy:
    def test_missing_layer_falls_back_to_default(self):
        """Layer not explicitly in policy should return the default config."""
        default = {"mode": "cartesian", "bits": 4, "group_size": 64}
        policy = LayerPolicy(default_config=default)
        policy.set_layer(0, mode="polar", bits=8, group_size=32)

        # Layer 0 is registered
        cfg_0 = policy.get_config(0)
        assert cfg_0["mode"] == "polar"
        assert cfg_0["bits"] == 8
        assert cfg_0["group_size"] == 32

        # Layer 99 is not registered — should fall back
        cfg_99 = policy.get_config(99)
        assert cfg_99["mode"] == "cartesian"
        assert cfg_99["bits"] == 4
        assert cfg_99["group_size"] == 64

    def test_invalid_layer_rejected(self):
        """Non-integer layer IDs should raise ValueError."""
        policy = LayerPolicy(default_config={"mode": "cartesian", "bits": 4})

        with pytest.raises(ValueError, match="layer_id"):
            policy.get_config("not_an_int")

        with pytest.raises(ValueError, match="layer_id"):
            policy.set_layer("not_an_int", mode="polar", bits=8)

        with pytest.raises(ValueError, match="layer_id"):
            policy.get_config(-1)

    def test_invalid_mode_rejected(self):
        """Unknown quantization modes should raise ValueError."""
        policy = LayerPolicy(default_config={"mode": "cartesian", "bits": 4})

        with pytest.raises(ValueError, match="mode"):
            policy.set_layer(0, mode="unknown_mode", bits=8)

        with pytest.raises(ValueError, match="mode"):
            policy.set_layer(1, mode="", bits=4)

    def test_init_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown default mode"):
            LayerPolicy(default_config={"mode": "phantom"})

    def test_init_missing_mode_key_raises(self):
        with pytest.raises(ValueError, match="Unknown default mode"):
            LayerPolicy(default_config={"bits": 4})

    def test_set_layer_zero_is_valid(self):
        policy = LayerPolicy(default_config={"mode": "cartesian"})
        policy.set_layer(0, mode="polar", bits=8)
        assert policy.get_config(0)["mode"] == "polar"

    def test_set_layer_float_id_rejected(self):
        policy = LayerPolicy(default_config={"mode": "cartesian"})
        with pytest.raises(ValueError, match="layer_id"):
            policy.set_layer(0.0, mode="polar", bits=8)

    def test_get_config_returns_copy(self):
        """Mutating returned config must not affect internal state."""
        policy = LayerPolicy(
            default_config={"mode": "cartesian", "bits": 4}
        )
        cfg = policy.get_config(0)
        cfg["bits"] = 99
        assert policy.get_config(0)["bits"] == 4

    def test_set_layer_overwrites_existing(self):
        policy = LayerPolicy(default_config={"mode": "cartesian"})
        policy.set_layer(0, mode="polar", bits=8)
        policy.set_layer(0, mode="hybrid", bits=6)
        assert policy.get_config(0)["mode"] == "hybrid"
        assert policy.get_config(0)["bits"] == 6

    def test_policy_produces_deterministic_config(self):
        """Same layer ID should always return the same config object."""
        policy = LayerPolicy(
            default_config={"mode": "cartesian", "bits": 4, "group_size": 64}
        )
        policy.set_layer(2, mode="hybrid", bits=6, group_size=128)

        cfg_a = policy.get_config(2)
        cfg_b = policy.get_config(2)
        assert cfg_a == cfg_b
        assert cfg_a is not None
        assert cfg_a["mode"] == "hybrid"
        assert cfg_a["bits"] == 6

    def test_load_valid_policy(self, tmp_path):
        """Valid JSON policy should load correctly."""
        data = {
            "default": {"mode": "cartesian", "bits": 4, "group_size": 64},
            "layers": {
                "0": {"mode": "polar", "bits": 8, "group_size": 32},
                "1": {"mode": "hybrid", "bits": 6, "group_size": 64},
            },
        }
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(data))

        policy = load_policy(str(path))
        assert policy.get_config(0)["mode"] == "polar"
        assert policy.get_config(1)["bits"] == 6
        assert policy.get_config(99)["mode"] == "cartesian"

    def test_load_invalid_policy_raises(self, tmp_path):
        """Invalid policy structure should raise ValueError."""
        # Missing "default" key
        bad_data = {"layers": {"0": {"mode": "polar", "bits": 8}}}
        path = tmp_path / "bad_policy.json"
        path.write_text(json.dumps(bad_data))

        with pytest.raises(ValueError, match="default"):
            load_policy(str(path))

        # Non-dict "layers" value
        bad_data_2 = {"default": {"mode": "cartesian"}, "layers": "not_a_dict"}
        path_2 = tmp_path / "bad_policy2.json"
        path_2.write_text(json.dumps(bad_data_2))

        with pytest.raises(ValueError, match="layers"):
            load_policy(str(path_2))


# ------------------------------------------------------------------
# validate_layer_policy tests
# ------------------------------------------------------------------

class TestValidateLayerPolicy:
    def test_valid_policy_returns_empty(self):
        policy = {
            "default": {"mode": "k8_v5_gs64"},
            "layers": {"0": {"mode": "k8_v5_gs32"}},
        }
        errors = validate_layer_policy(policy)
        assert errors == []

    def test_missing_default_key(self):
        errors = validate_layer_policy({"layers": {}})
        assert any("default" in e for e in errors)

    def test_non_dict_default(self):
        errors = validate_layer_policy({"default": "not_a_dict"})
        assert any("default" in e for e in errors)

    def test_unknown_default_mode(self):
        errors = validate_layer_policy({"default": {"mode": "unknown"}})
        assert any("Unknown default mode" in e for e in errors)

    def test_non_dict_layers(self):
        errors = validate_layer_policy(
            {"default": {"mode": "k8_v5_gs64"}, "layers": []}
        )
        assert any("layers" in e for e in errors)

    def test_invalid_layer_id(self):
        policy = {
            "default": {"mode": "k8_v5_gs64"},
            "layers": {"not_a_number": {"mode": "k8_v5_gs32"}},
        }
        errors = validate_layer_policy(policy)
        assert any("Invalid layer ID" in e for e in errors)

    def test_negative_layer_id(self):
        policy = {
            "default": {"mode": "k8_v5_gs64"},
            "layers": {"-1": {"mode": "k8_v5_gs32"}},
        }
        errors = validate_layer_policy(policy)
        assert any("Negative layer ID" in e for e in errors)

    def test_non_dict_layer_config(self):
        policy = {
            "default": {"mode": "k8_v5_gs64"},
            "layers": {"0": "not_a_dict"},
        }
        errors = validate_layer_policy(policy)
        assert any("config must be a dict" in e for e in errors)

    def test_unknown_layer_mode(self):
        policy = {
            "default": {"mode": "k8_v5_gs64"},
            "layers": {"0": {"mode": "super_new"}},
        }
        errors = validate_layer_policy(policy)
        assert any("unknown mode" in e for e in errors)

    def test_known_modes_coverage(self):
        assert "baseline_fp16" in KNOWN_MODES
        assert "k8_v5_gs64" in KNOWN_MODES
        assert "adaptive" in KNOWN_MODES
        assert "experimental_hybrid" in KNOWN_MODES
        assert "turbo_polar" in KNOWN_MODES

    def test_invalid_policy_type(self):
        errors = validate_layer_policy([])
        assert errors == ["Policy must be a dict"]

    def test_truthy_non_dict_layers(self):
        """Truthy non-dict layers (e.g. int) should not crash."""
        errors = validate_layer_policy(
            {"default": {"mode": "k8_v5_gs64"}, "layers": 1}
        )
        assert any("layers" in e for e in errors)

    def test_truthy_non_dict_layers_string(self):
        errors = validate_layer_policy(
            {"default": {"mode": "k8_v5_gs64"}, "layers": "foo"}
        )
        assert any("layers" in e for e in errors)

    def test_default_missing_mode(self):
        errors = validate_layer_policy({"default": {}})
        assert any("Unknown default mode" in e for e in errors)

    def test_layer_missing_mode(self):
        policy = {
            "default": {"mode": "k8_v5_gs64"},
            "layers": {"0": {}},
        }
        errors = validate_layer_policy(policy)
        assert any("unknown mode" in e for e in errors)

    def test_no_layers_key_is_valid(self):
        errors = validate_layer_policy({"default": {"mode": "k8_v5_gs64"}})
        assert errors == []

    def test_layers_none_is_valid(self):
        errors = validate_layer_policy(
            {"default": {"mode": "k8_v5_gs64"}, "layers": None}
        )
        assert errors == []

    def test_negative_layer_id_with_bad_mode(self):
        """Negative layer ID should produce exactly one error, not cascade."""
        policy = {
            "default": {"mode": "k8_v5_gs64"},
            "layers": {"-1": {"mode": "bad_mode"}},
        }
        errors = validate_layer_policy(policy)
        negative_errors = [e for e in errors if "Negative layer ID" in e]
        mode_errors = [e for e in errors if "unknown mode" in e]
        assert len(negative_errors) == 1
        assert len(mode_errors) == 0

    def test_bits_zero_rejected(self):
        """bits=0 is invalid and must be rejected."""
        policy = {
            "default": {"mode": "k8_v5_gs64", "bits": 0},
        }
        errors = validate_layer_policy(policy)
        bit_errors = [e for e in errors if "8-bit pack limit" in e]
        assert len(bit_errors) == 1

    def test_bits_nine_rejected(self):
        """bits >= 9 should trigger an error about bit-packing."""
        policy = {
            "default": {"mode": "k8_v5_gs64", "bits": 9},
        }
        errors = validate_layer_policy(policy)
        assert any("exceeds 8-bit pack limit" in e for e in errors)
        assert any("cannot be memory-optimized" in e for e in errors)

    def test_bits_eight_ok(self):
        """bits=8 is inside the true bitpack range; no error expected."""
        policy = {
            "default": {"mode": "k8_v5_gs64", "bits": 8},
        }
        errors = validate_layer_policy(policy)
        bit_errors = [e for e in errors if "8-bit pack limit" in e]
        assert len(bit_errors) == 0

    def test_cartesian_bits_rejected(self):
        """cartesian_bits > 8 should trigger bit-packing error."""
        policy = {
            "default": {
                "mode": "k8_v5_gs64",
                "cartesian_bits": 10,
            },
        }
        errors = validate_layer_policy(policy)
        assert any("exceeds 8-bit pack limit" in e for e in errors)

    def test_bits_one_rejected(self):
        """bits=1 is below 2-bit pack threshold."""
        policy = {
            "default": {"mode": "k8_v5_gs64", "bits": 1},
        }
        errors = validate_layer_policy(policy)
        assert any("exceeds 8-bit pack limit" in e for e in errors)

    def test_bits_two_ok(self):
        """bits=2 is the lower bound of true bit-packing."""
        policy = {
            "default": {"mode": "k8_v5_gs64", "bits": 2},
        }
        errors = validate_layer_policy(policy)
        bit_errors = [e for e in errors if "8-bit pack limit" in e]
        assert len(bit_errors) == 0


# ------------------------------------------------------------------
# load_policy tests
# ------------------------------------------------------------------

class TestLoadPolicy:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_policy("/nonexistent/path/policy.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            load_policy(str(path))

    def test_json_not_dict_raises(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="dict"):
            load_policy(str(path))

    def test_load_with_path_object(self, tmp_path):
        data = {"default": {"mode": "cartesian"}}
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(data))
        policy = load_policy(path)
        assert policy.get_config(0)["mode"] == "cartesian"

    def test_load_missing_default_raises(self, tmp_path):
        path = tmp_path / "no_default.json"
        path.write_text(json.dumps({"layers": {}}))
        with pytest.raises(ValueError, match="default"):
            load_policy(str(path))

    def test_load_non_dict_default_raises(self, tmp_path):
        path = tmp_path / "bad_default.json"
        path.write_text(json.dumps({"default": "bad"}))
        with pytest.raises(ValueError, match="default"):
            load_policy(str(path))

    def test_load_non_dict_layers_raises(self, tmp_path):
        path = tmp_path / "bad_layers.json"
        path.write_text(
            json.dumps(
                {"default": {"mode": "cartesian"}, "layers": []}
            )
        )
        with pytest.raises(ValueError, match="layers"):
            load_policy(str(path))

    def test_load_invalid_layer_id_raises(self, tmp_path):
        path = tmp_path / "bad_layer_id.json"
        path.write_text(
            json.dumps(
                {
                    "default": {"mode": "cartesian"},
                    "layers": {"abc": {"mode": "polar"}},
                }
            )
        )
        with pytest.raises(ValueError, match="Invalid layer ID"):
            load_policy(str(path))

    def test_load_negative_layer_id_raises(self, tmp_path):
        path = tmp_path / "neg_layer.json"
        path.write_text(
            json.dumps(
                {
                    "default": {"mode": "cartesian"},
                    "layers": {"-1": {"mode": "polar"}},
                }
            )
        )
        with pytest.raises(ValueError, match="Negative layer ID"):
            load_policy(str(path))

    def test_load_non_dict_layer_config_raises(self, tmp_path):
        path = tmp_path / "bad_cfg.json"
        path.write_text(
            json.dumps(
                {
                    "default": {"mode": "cartesian"},
                    "layers": {"0": "bad"},
                }
            )
        )
        with pytest.raises(ValueError, match="config"):
            load_policy(str(path))

    def test_load_unknown_layer_mode_raises(self, tmp_path):
        path = tmp_path / "bad_mode.json"
        path.write_text(
            json.dumps(
                {
                    "default": {"mode": "cartesian"},
                    "layers": {"0": {"mode": "unknown"}},
                }
            )
        )
        with pytest.raises(ValueError, match="mode"):
            load_policy(str(path))

    def test_all_known_modes_accepted(self):
        for mode in KNOWN_MODES:
            policy = LayerPolicy(default_config={"mode": mode})
            assert policy.get_config(0)["mode"] == mode
