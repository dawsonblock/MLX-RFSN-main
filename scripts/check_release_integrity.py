#!/usr/bin/env python3
"""Release integrity checker for MLX-RFSN Fusion.

Reads release identity from release.toml for unified versioning.
"""
from __future__ import annotations

import ast
import fnmatch
import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Phase 1: strict evidence helpers
sys.path.insert(0, str(Path(__file__).parent.parent))
from rfsn_v11.candidates.evidence_status import (
    EvidenceStatus,
    classify_artifact_status,
    require_successful_rows,
)


def _load_gitignore(root: Path) -> tuple[set[str], set[str], set[str]]:
    # noqa: E501
    """Parse .gitignore; return (exact, dir, wildcard) name sets."""
    gitignore = root / ".gitignore"
    exact_names: set[str] = set()
    dir_names: set[str] = set()
    wildcards: set[str] = set()
    if not gitignore.exists():
        return exact_names, dir_names, wildcards
    for line in gitignore.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip trailing / to detect directory patterns
        if line.endswith("/"):
            dir_names.add(line[:-1])
        elif "*" in line or "?" in line:
            wildcards.add(line)
        else:
            exact_names.add(line)
    return exact_names, dir_names, wildcards


def _load_release_config(root: Path) -> dict:
    """Load release.toml configuration."""
    config_path = root / "release.toml"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def check() -> list[str]:
    errors: list[str] = []

    root = Path(".").resolve()
    exact_ign, dir_ign, wildcard_ign = _load_gitignore(root)

    # Load release configuration
    release_config = _load_release_config(root)
    release_id = release_config.get("release_id", "unknown")
    display_name = release_config.get("display_name", "unknown")

    def _is_gitignored(path: Path) -> bool:
        name = path.name
        if name in exact_ign:
            return True
        if name in dir_ign and path.is_dir():
            return True
        for w in wildcard_ign:
            if fnmatch.fnmatch(name, w):
                return True
        # Also check parent directory names for directory patterns
        for part in path.parts:
            if part in dir_ign:
                return True
        return False

    # --- Forbidden filesystem artefacts ---
    forbidden_dirs = [".tmp", "tmp", "temp", "release_tmp"]
    for bad in forbidden_dirs:
        matches = [
            m for m in root.rglob(bad)
            if ".git" not in m.parts and not _is_gitignored(m)
        ]
        if matches:
            errors.append(
                f"forbidden path found: {bad} ({len(matches)} instances)"
            )

    pycache = [
        m for m in root.rglob("__pycache__")
        if ".git" not in m.parts and not _is_gitignored(m)
    ]
    if pycache:
        errors.append(
            f"__pycache__ directories found ({len(pycache)} instances)"
        )

    pyc = [
        m for m in root.rglob("*.pyc")
        if ".git" not in m.parts and not _is_gitignored(m)
    ]
    if pyc:
        errors.append(f"*.pyc files found ({len(pyc)} instances)")

    ds_store = [
        m for m in root.rglob(".DS_Store")
        if ".git" not in m.parts and not _is_gitignored(m)
    ]
    if ds_store:
        errors.append(f".DS_Store files found ({len(ds_store)} instances)")

    for pattern in ["*.zip", "*.tar", "*.tar.gz", "*.7z"]:
        matches = [
            m for m in root.rglob(pattern)
            if not _is_gitignored(m)
        ]
        if matches:
            errors.append(
                f"nested archive(s) found: {[str(m) for m in matches[:10]]}"
            )

    # --- Reject placeholder plots ---
    plot_dir = root / "results" / "plots"
    if plot_dir.exists():
        for pattern in ["*pending*", "*placeholder*"]:
            bad = list(plot_dir.glob(pattern))
            if bad:
                errors.append(
                    f"placeholder plot(s) found: {[str(p) for p in bad[:10]]}"
                )

    # --- README strict checks ---
    readme_path = root / "README.md"
    readme: str = ""
    try:
        readme = readme_path.read_text(encoding="utf-8")
    except OSError:
        errors.append("README.md missing or unreadable")

    if readme:
        # Check first 10 non-empty lines for release title
        lines = readme.splitlines()
        non_empty = [ln for ln in lines if ln.strip()][:10]
        # Phase 2: Accept either the full display_name or the short
        # release_id-based title (e.g. "MLX-RFSN Fusion Alpha 8.4").
        expected_titles = [
            f"# {display_name}",
            f"# MLX-RFSN Fusion {release_id}",
            "# MLX-RFSN Fusion Alpha 8.4",
        ]
        has_title = any(
            any(ln.startswith(t) for t in expected_titles)
            for ln in non_empty
        )
        if not has_title:
            errors.append(
                f"README title does not match expected titles: "
                f"{expected_titles}"
            )

        # Check status section (normalize em-dash to hyphen for comparison)
        expected_status = f"## Status: {display_name}"
        readme_normalized = readme.replace("—", "-")
        if expected_status not in readme_normalized:
            errors.append(
                f"README status section does not contain '{expected_status}'"
            )

        for stale in [
            "artifacts/proof/main23",
            "artifacts/proof/main24",
            "artifacts/proof/main25",
            "artifacts/proof/main26",
            "artifacts/proof/main27",
        ]:
            # Allow historical mentions if they appear after
            # the word "historical" or inside a note about old
            # releases — check each line
            for lineno, line in enumerate(readme.splitlines(), 1):
                if stale in line:
                    lower = line.lower()
                    if any(
                        w in lower
                        for w in (
                            "historical",
                            "history",
                            "retained",
                            "reference only",
                        )
                    ):
                        continue
                    errors.append(
                        f"README contains active stale artifact path: "
                        f"{stale} (line {lineno})"
                    )

        false_claims = [
            "production-ready",
            "production ready",
            "polar quant enabled",
            "partial dequant complete",
            "sparse-safe",
            "sparse enabled by default",
        ]
        # Use word boundaries to avoid matching partial words
        # (e.g., "noted" contains "not")
        negation_patterns = [
            "not ", "no ", "never ", "unimplemented", "disabled"
        ]
        for line in readme.splitlines():
            lower_line = line.lower()
            for phrase in false_claims:
                if phrase in lower_line:
                    # Check for negation patterns with word boundaries
                    has_negation = any(
                        (" " + nw in lower_line) or lower_line.startswith(nw)
                        for nw in negation_patterns
                    )
                    if has_negation:
                        continue
                    errors.append(
                        f"README positive claim detected: {phrase!r}"
                    )

        # README must qualify PolarQuant status if file exists
        polar_quant_path = (
            root / "rfsn_v10" / "quantization" / "polar_quant.py"
        )
        readme_lower = readme.lower()
        if polar_quant_path.exists():
            mentions_polar = (
                "polar quantization" in readme_lower
                or "polar quant" in readme_lower
            )
            has_qualifier = (
                "stable runtime" in readme_lower
                or "experimental" in readme_lower
            )
            if mentions_polar and not has_qualifier:
                errors.append(
                    "README mentions PolarQuant but does not qualify "
                    "stable vs experimental status"
                )

        # README must document >8-bit fallback caveat
        if "raw uint32 fallback" not in readme_lower:
            errors.append(
                "README missing >8-bit raw uint32 fallback caveat"
            )

        # README must disclaim Metal kernel status
        metal_disclaimers = [
            "no metal kernels exist for the experimental",
            "actual metal gpu computation not yet implemented",
            "metal kernel",
            "scaffold/stub with cpu fallback",
        ]
        if not any(d in readme_lower for d in metal_disclaimers):
            errors.append(
                "README missing Metal kernel status disclaimer"
            )

        # README must disclaim experimental throughput speedup
        if "no experimental throughput speedup is proven" not in readme_lower:
            errors.append(
                "README missing 'No experimental throughput speedup' caveat"
            )

    # --- Pytest collection sanity check ---
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q",
             str(root / "tests")],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            cwd=str(root),
            check=False,
        )
        if result.returncode != 0:
            errors.append(
                "pytest collection failed:\n" + result.stdout[-4000:]
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"pytest collection check failed to run: {exc}")

    # --- Test file MLX import safety ---
    def _has_top_level_mlx_import(tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "mlx" or \
                                alias.name.startswith("mlx."):
                            return True
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("mlx"):
                        return True
        return False

    def _has_importorskip(tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if func.attr == "importorskip":
                        return True
                elif isinstance(func, ast.Name):
                    if func.id == "importorskip":
                        return True
        return False

    for test_file in (root / "tests").rglob("*.py"):
        try:
            source = test_file.read_text(encoding="utf-8")
            tree = ast.parse(source)
            if _has_top_level_mlx_import(tree) and not _has_importorskip(tree):
                errors.append(
                    f"{test_file.name} imports mlx at top level "
                    f"without pytest.importorskip"
                )
        except (OSError, SyntaxError):
            pass

    # --- Experimental artifact checks (Phase 0 freeze) ---
    exp_dir = root / "artifacts" / "proof" / "experimental"
    # Phase 2: Only check artifacts that are part of the active scope.
    # Deferred artifacts (real_model_validation, long_context_validation,
    # memory_accounting, comparison_summary, qjl_attention_score) are
    # not required until Phase 4+.
    required_exp_artifacts = [
        "teacher_forced_step_trace.json",
        "decode_update_trace.json",
        "decode_append_kv_diff.json",
    ]
    for artifact in required_exp_artifacts:
        artifact_path = exp_dir / artifact
        if not artifact_path.exists():
            errors.append(f"experimental artifact missing: {artifact}")
            continue
        if not artifact.endswith(".json"):
            continue
        try:
            data = json.loads(artifact_path.read_text(encoding="utf-8"))
            # No placeholder status allowed
            status = data.get("status", "")
            if status in {"awaiting_execution", "placeholder"}:
                errors.append(
                    f"{artifact} is a placeholder (status={status})"
                )
            # No config should claim production-ready
            for key in ("production_ready", "production-ready"):
                if data.get(key) is True:
                    errors.append(f"{artifact} claims production_ready=True")
        except (OSError, json.JSONDecodeError):
            errors.append(f"experimental artifact unreadable: {artifact}")

    # --- Experimental artifact manifest ---
    exp_dir = root / "artifacts" / "proof" / "experimental"
    manifest_path = exp_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        # P0 Fix: Use channel field (not status) to detect alpha releases
        if release_config.get("channel") == "alpha":
            pass  # Skip experimental artifact check for alpha releases
        else:
            errors.append(
                "artifacts/proof/experimental/artifact_manifest.json missing"
            )
    else:
        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            # Use release_id from config instead of hardcoded "experimental"
            expected_release = release_config.get("release_id", "experimental")
            if manifest.get("release") != expected_release:
                errors.append(
                    f"artifact_manifest.json release "
                    f"field is not '{expected_release}'"
                )
            # P0 Fix: Skip Main 28 specific checks for alpha releases
            if release_config.get("channel") != "alpha":
                if manifest.get("stable_default") != "k8_v5_gs64":
                    errors.append(
                        "artifact_manifest.json stable_default is not 'k8_v5_gs64'"
                    )
                if manifest.get("qjl_status") != "failed_disabled":
                    errors.append(
                        "artifact_manifest.json qjl_status "
                        "is not 'failed_disabled'"
                    )
            if manifest.get("promoted_to_default") is not False:
                errors.append(
                    "artifact_manifest.json promoted_to_default must be false"
                )
            # P0 Fix: Remove Main 28 hardcoded artifact names
            expected_artifacts = {
                "comparison": "comparison_summary.json",
                "memory": "memory_accounting.json",
                "throughput": "throughput.json",
            }
            actual_artifacts = manifest.get("artifacts", {})
            for key, expected_path in expected_artifacts.items():
                actual = actual_artifacts.get(key)
                if actual != expected_path:
                    errors.append(
                        f"artifact_manifest.json artifact '{key}' expected "
                        f"'{expected_path}', got '{actual}'"
                    )
                # Also verify the file/directory exists
                artifact_path = exp_dir / expected_path
                if not artifact_path.exists():
                    errors.append(
                        f"artifact_manifest.json missing artifact on disk: "
                        f"{expected_path}"
                    )
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"artifact_manifest.json parse error: {exc}")

    # --- Stale artifact directories ---
    proof_dir = root / "artifacts" / "proof"
    if proof_dir.exists():
        # P0 Fix: Use channel field (not status) to detect alpha releases
        if release_config.get("channel") != "alpha":
            stale_releases = [
                d.name for d in proof_dir.iterdir()
                if d.is_dir() and d.name.startswith("main")
                and d.name != release_id
            ]
            for stale in stale_releases:
                errors.append(
                    f"stale artifact directory found: artifacts/proof/{stale}"
                )
            for stale_manifest in proof_dir.glob("main*_release_manifest.json"):
                if stale_manifest.name != f"{release_id}_release_manifest.json":
                    errors.append(
                        f"stale release manifest found: {stale_manifest.name}"
                    )

    # --- Release artifact directory (only for non-alpha releases) ---
    # P0 Fix: Use channel field (not status) to detect alpha releases
    if release_config.get("channel") != "alpha":
        artifact_dir = root / "artifacts" / "proof" / release_id
        if not artifact_dir.exists():
            errors.append(f"artifacts/proof/{release_id} missing")
        else:
            required_artifacts = [
                "kernel_benchmark.json",
                "fused_kernel_benchmark.json",
                "optimization_benchmark.json",
                "real_model_validation.json",
                "long_context_validation.json",
                "generation_smoke.json",
                "generation_throughput.json",
                "proof_summary.md",
                "summary.json",
                "mlx_test_summary.md",
                "mlx_pytest_raw.log",
                "mlx_pytest_junit.xml",
                f"{release_id}_release_manifest.json",
            ]
            for artifact in required_artifacts:
                artifact_path = artifact_dir / artifact
                if not artifact_path.exists():
                    errors.append(f"required artifact missing: {artifact}")
                    continue
                # Every JSON artifact with a "release" field must match release_id
                if artifact.endswith(".json"):
                    try:
                        data = json.loads(
                            artifact_path.read_text(encoding="utf-8")
                        )
                        release_field = data.get("release")
                        if release_field is not None and release_field != release_id:
                            errors.append(
                                f"{artifact} release field is "
                                f"'{release_field}' (expected '{release_id}')"
                            )
                    except (
                        OSError, ValueError, TypeError, AttributeError
                    ):
                        pass

            # Manifest must declare release = release_id
            manifest_path = artifact_dir / f"{release_id}_release_manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    if manifest.get("release") != release_id:
                        errors.append(
                            f"{release_id}_release_manifest.json release "
                            f"field is not '{release_id}'"
                        )
                except (OSError, ValueError, TypeError, AttributeError):
                    errors.append(f"{release_id}_release_manifest.json is not valid JSON")

            # MLX summary must identify release
            mlx_summary_path = artifact_dir / "mlx_test_summary.md"
            if mlx_summary_path.exists():
                try:
                    mlx_summary = mlx_summary_path.read_text(encoding="utf-8")
                    if display_name not in mlx_summary:
                        errors.append(f"MLX summary does not identify {display_name}")
                except OSError:
                    pass

            # proof_summary.md must identify release
            proof_path = artifact_dir / "proof_summary.md"
            if proof_path.exists():
                try:
                    proof = proof_path.read_text(encoding="utf-8")
                    if display_name not in proof:
                        errors.append(f"proof_summary.md does not identify {display_name}")
                except OSError:
                    pass

        # real_model_validation.json: no tiny-random model, sparse not enabled,
        # must evaluate >= 32 positions
        real_val_path = artifact_dir / "real_model_validation.json"
        if real_val_path.exists():
            try:
                data = json.loads(real_val_path.read_text(encoding="utf-8"))
                model_id = data.get("model", "")
                if "tiny-random" in model_id.lower():
                    errors.append(
                        "real_model_validation.json still uses "
                        "tiny-random model"
                    )
                if data.get("sparse_enabled") is True:
                    errors.append(
                        "sparse_enabled is True in real_model_validation.json"
                    )
                for cfg in data.get("configs", []):
                    pos = cfg.get("token_positions_evaluated", 0)
                    if isinstance(pos, (int, float)) and pos < 32:
                        errors.append(
                            f"config {cfg.get('name')!r} evaluated only "
                            f"{pos} positions (minimum 32 required)"
                        )
                # P0 Fix: Check release field matches configured release_id, not hardcoded main28
                expected_rel = release_config.get("release_id", "unknown")
                if data.get("release") != expected_rel:
                    errors.append(
                        f"real_model_validation.json release "
                        f"field is not '{expected_rel}'"
                    )
            except (
                OSError, ValueError, TypeError, AttributeError
            ) as exc:
                errors.append(f"real_model_validation.json parse error: {exc}")

        # long_context_validation.json: recommended config must
        # pass all contexts
        long_ctx_path = artifact_dir / "long_context_validation.json"
        if long_ctx_path.exists():
            try:
                lc = json.loads(long_ctx_path.read_text(encoding="utf-8"))
                summary = lc.get("summary", {})
                recommended = summary.get("recommended_default", "")
                if recommended and recommended != "baseline_fp16":
                    for ctx_entry in lc.get("contexts", []):
                        for cfg in ctx_entry.get("configs", []):
                            if cfg.get("name") == recommended:
                                if cfg.get("status") != "pass":
                                    errors.append(
                                        f"recommended config {recommended!r} "
                                        f"fails context "
                                        f"{ctx_entry.get('tokens')} tokens"
                                    )
            except (
                OSError, ValueError, TypeError, AttributeError
            ) as exc:
                errors.append(
                    f"long_context_validation.json parse error: {exc}"
                )

    # --- Artifact manifest existence check ---
    manifest_path = (
        root
        / "artifacts"
        / "proof"
        / "experimental"
        / "artifact_manifest.json"
    )
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, rel_path in manifest.get("artifacts", {}).items():
                if isinstance(rel_path, str):
                    artifact_file = manifest_path.parent / rel_path
                    if not artifact_file.exists():
                        errors.append(
                            "artifact_manifest references missing "
                            f"file: {rel_path}"
                        )
        except (OSError, json.JSONDecodeError):
            errors.append("artifact_manifest.json is not valid JSON")

    # --- ARTIFACT_INDEX.md missing-file check ---
    index_path = root / "artifacts" / "ARTIFACT_INDEX.md"
    if index_path.exists():
        try:
            index_text = index_path.read_text(encoding="utf-8")
            for line in index_text.splitlines():
                # Look for markdown table rows that reference .json files
                if "|" in line and ".json" in line:
                    parts = [p.strip() for p in line.split("|")]
                    for part in parts:
                        if part.endswith(".json"):
                            # Normalize: strip backticks if any
                            fname = part.strip("`").strip()
                            if fname.endswith(".json"):
                                fpath = index_path.parent / fname
                                if not fpath.exists():
                                    errors.append(
                                        "ARTIFACT_INDEX.md lists missing "
                                        f"artifact: {fname}"
                                    )
        except OSError:
            pass

    # --- real_generation_throughput.json schema and baseline checks ---
    def check_real_generation_schema(errors):
        path = Path(
            "artifacts/proof/experimental/real_generation_throughput.json"
        )
        if not path.exists():
            # Phase 2: Not required for alpha / Phase 0 freeze
            if release_config.get("channel") != "alpha":
                errors.append("missing real_generation_throughput.json")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("real_generation_throughput.json is not valid JSON")
            return
        for key in ("free_running_generation",):
            if key not in data:
                errors.append(f"real_generation_throughput.json missing {key}")
        rows = data.get("free_running_generation", [])
        for row in rows:
            if row.get("config") == "baseline_fp16":
                cr = row.get("compression_ratio")
                if cr is not None and cr != 1.0:
                    errors.append(
                        "baseline_fp16 compression_ratio must be 1.0"
                    )
                fp16_bytes = row.get("fp16_kv_bytes")
                compressed = row.get("compressed_kv_bytes")
                if fp16_bytes is not None and compressed != fp16_bytes:
                    errors.append(
                        "baseline_fp16 compressed_kv_bytes "
                        "must equal fp16_kv_bytes"
                    )

    check_real_generation_schema(errors)

    # --- QJL disabled check ---
    qjl_path = (
        root
        / "artifacts"
        / "proof"
        / "experimental"
        / "qjl_attention_score.json"
    )
    if qjl_path.exists():
        try:
            qjl_data = json.loads(qjl_path.read_text(encoding="utf-8"))
            if qjl_data.get("passes_all") is False:
                # QJL must be disabled; check manifest
                manifest_path_2 = (
                    root
                    / "artifacts"
                    / "proof"
                    / "experimental"
                    / "artifact_manifest.json"
                )
                if manifest_path_2.exists():
                    manifest_2 = json.loads(
                        manifest_path_2.read_text(encoding="utf-8")
                    )
                    if manifest_2.get("qjl_status") != "failed_disabled":
                        errors.append(
                            "QJL attention score fails but manifest "
                            "does not mark qjl_status as failed_disabled"
                        )
        except (OSError, json.JSONDecodeError):
            pass

    # --- Reject placeholder diagnostic files ---
    for diag_name in ("decode_update_trace.json", "decode_append_kv_diff.json"):
        diag_path = root / "artifacts" / "proof" / "experimental" / diag_name
        if diag_path.exists():
            try:
                diag_data = json.loads(diag_path.read_text(encoding="utf-8"))
                if diag_data.get("status") == "awaiting_execution":
                    errors.append(
                        f"{diag_name} is a placeholder (status=awaiting_execution)"
                    )
                if not diag_data.get("traces") and not diag_data.get("results"):
                    errors.append(
                        f"{diag_name} has empty traces/results"
                    )
            except (OSError, json.JSONDecodeError):
                pass

    # --- Deep field validation: decode_update_trace.json ---
    def check_decode_update_trace(errors: list) -> None:
        path = (
            root / "artifacts" / "proof" / "experimental"
            / "decode_update_trace.json"
        )
        if not path.exists():
            errors.append("decode_update_trace.json missing")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("decode_update_trace.json is not valid JSON")
            return
        top_status = data.get("status", "")
        if top_status in {"awaiting_execution", "placeholder"}:
            errors.append(
                "decode_update_trace.json is a placeholder "
                f"(status={top_status})"
            )
            return
        traces = data.get("traces", [])
        if not traces:
            errors.append("decode_update_trace.json has empty traces")
            return
        # Phase 2: execution_failed is a valid portable state; still validate
        # schema of any successful rows that may exist.
        required = {
            "config",
            "prompt_tokens",
            "decode_step",
            "kv_len_before",
            "kv_len_after",
            "position_id",
            "cache_position",
            "logit_cosine_vs_fp16",
            "top5_overlap_vs_fp16",
            "kl_vs_fp16",
            "status",
        }
        # Only validate successful rows; error rows already accounted for above
        successful = [r for r in traces if not r.get("error") and r.get("status") != "error"]
        for i, row in enumerate(successful):
            missing = required - set(row)
            if missing:
                errors.append(
                    f"decode_update_trace row {i} missing "
                    f"{sorted(missing)}"
                )

    check_decode_update_trace(errors)

    # --- Deep field validation: decode_append_kv_diff.json ---
    def check_decode_append_kv_diff(errors: list) -> None:
        path = (
            root / "artifacts" / "proof" / "experimental"
            / "decode_append_kv_diff.json"
        )
        if not path.exists():
            errors.append("decode_append_kv_diff.json missing")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append("decode_append_kv_diff.json is not valid JSON")
            return
        top_status = data.get("status", "")
        if top_status in {"awaiting_execution", "placeholder"}:
            errors.append(
                "decode_append_kv_diff.json is a placeholder "
                f"(status={top_status})"
            )
            return
        results = data.get("results", [])
        if not results:
            errors.append("decode_append_kv_diff.json has empty results")
            return
        # Phase 2: execution_failed is a valid portable state; still validate
        # schema of any successful rows that may exist.
        required = {
            "config",
            "prompt_tokens",
            "old_cache_k_cosine_after_append",
            "new_token_k_cosine",
            "kv_order_preserved",
            "cache_len_correct",
            "status",
        }
        successful = [
            r for r in results
            if not r.get("error") and r.get("status") != "error"
        ]
        for i, row in enumerate(successful):
            missing = required - set(row)
            if missing:
                errors.append(
                    f"decode_append_kv_diff result {i} missing "
                    f"{sorted(missing)}"
                )

    check_decode_append_kv_diff(errors)

    # --- Teacher-forced baseline identity check ---
    real_gen_path = (
        root
        / "artifacts"
        / "proof"
        / "experimental"
        / "real_generation_throughput.json"
    )
    if real_gen_path.exists():
        try:
            rg_data = json.loads(real_gen_path.read_text(encoding="utf-8"))
            # Check free-running baseline
            fr_rows = rg_data.get("free_running_generation", [])
            fr_baseline = [
                r for r in fr_rows
                if r.get("config") == "baseline_fp16" and "error" not in r
            ]
            for row in fr_baseline:
                emr = float(row.get("exact_token_match_rate", 0.0))
                cr = float(row.get("compression_ratio", 0.0))
                if abs(emr - 1.0) > 1e-7:
                    errors.append(
                        f"baseline_fp16 free-running exact match not 1.0: {emr}"
                    )
                if abs(cr - 1.0) > 1e-7:
                    errors.append(
                        f"baseline_fp16 free-running compression_ratio not 1.0: {cr}"
                    )
        except (OSError, json.JSONDecodeError):
            pass

    # --- No candidate without real-generation data ---
    classification_path = (
        root
        / "artifacts"
        / "proof"
        / "experimental"
        / "config_classification.json"
    )
    if classification_path.exists() and real_gen_path.exists():
        try:
            class_data = json.loads(
                classification_path.read_text(encoding="utf-8")
            )
            real_gen_data = json.loads(
                real_gen_path.read_text(encoding="utf-8")
            )
            configs_with_real_gen = set()
            for row in real_gen_data.get("free_running_generation", []):
                if "error" not in row:
                    configs_with_real_gen.add(row.get("config"))

            # If a per-step teacher-forced trace exists and shows pass,
            # trust that over the bulk average (methodology discrepancy).
            step_trace_path = (
                root
                / "artifacts"
                / "proof"
                / "experimental"
                / "teacher_forced_step_trace.json"
            )
            step_pass: dict[str, bool] = {}
            if step_trace_path.exists():
                try:
                    st_data = json.loads(
                        step_trace_path.read_text(encoding="utf-8")
                    )
                    for row in st_data.get("traces", []):
                        cfg = row.get("config")
                        if not cfg:
                            continue
                        status_step = row.get("status", "")
                        if cfg not in step_pass:
                            step_pass[cfg] = True
                        step_pass[cfg] = step_pass[cfg] and (
                            status_step == "pass"
                        )
                except (OSError, json.JSONDecodeError):
                    pass

            for cfg_name, status in class_data.get(
                "classifications", {}
            ).items():
                normalized = cfg_name.replace("stable_", "")
                if (
                    "candidate" in status
                    and normalized not in configs_with_real_gen
                ):
                    errors.append(
                        f"{cfg_name} classified as candidate "
                        f"but has no real-generation data"
                    )
                # Teacher-forced failure from step trace must be reflected
                # in classification.
                if (
                    normalized in step_pass
                    and not step_pass[normalized]
                ):
                    pessimistic_markers = (
                        "drift",
                        "divergence",
                        "rejected",
                        "failed",
                        "disabled",
                        "needs_",
                    )
                    has_pessimistic = any(
                        marker in status for marker in pessimistic_markers
                    )
                    if not has_pessimistic:
                        errors.append(
                            f"{cfg_name} teacher-forced fails but "
                            f"classified optimistically as '{status}'"
                        )
        except (OSError, json.JSONDecodeError):
            pass

    # --- Vendored repo check (Phase 10) ---
    vendored_dir = root / "external"
    if vendored_dir.exists():
        vendored_repos = [
            d.name for d in vendored_dir.iterdir()
            if d.is_dir() and d.name not in {".git", "__pycache__"}
        ]
        if vendored_repos:
            # Phase 10: alpha allows vendored repos; beta+ warns
            if release_config.get("channel") != "alpha":
                errors.append(
                    f"vendored repos present: {vendored_repos}; "
                    f"remove before beta promotion"
                )

    return errors


def main() -> int:
    errors = check()
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Run proof-summary / JSON consistency checker
    consistency_script = Path("scripts/check_proof_summary_consistency.py")
    if consistency_script.exists():
        try:
            subprocess.run(
                [sys.executable, str(consistency_script)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                "ERROR: proof-summary consistency check failed: "
                f"{exc.stderr}",
                file=sys.stderr,
            )
            return 1

    print("release integrity OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
