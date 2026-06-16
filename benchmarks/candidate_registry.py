"""RFSN candidate registry.

Fix #8: One authoritative candidate registry.
Phase 0 Scope Freeze (2026-06-15): The default registry contains ONLY
three canonical candidates required for K8/V8 GS64 validation:

  1. dense_mlx_baseline              — FP16 dense control
  2. mlx_lm_8bit_kv                  — MLX-LM built-in 8-bit KV control
  3. rfsn_direct_packed_k8v8_gs64    — RFSN direct-packed K8/V8 GS64

All other candidates (Polar, TurboQuant, QJL, K8/V5, K8/V6, K16/V8,
K8/V16, K16/V16, SnapKV, SparseJL, etc.) remain in the repository as
archived experiments but are EXCLUDED from the default registry.

They can be accessed via build_experimental_registry() or
get_experimental_registry() for explicit research work.

Maps canonical candidate names to instantiated candidate objects.
All imports are lazy — missing optional dependencies only raise
when the specific candidate is requested.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Ensure project root is importable
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

if TYPE_CHECKING:
    from benchmarks.candidates.base_candidate import BenchmarkCandidate


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class CandidateRegistry:
    """One authoritative candidate registry for all benchmark candidates.

    Fix #8: This is the single source of truth for candidate definitions.
    The shootout and all other code must use this registry, not hardcoded construction.

    Methods:
    - declared(): All declared candidates (no dependency checks)
    - available(environment): Candidates that can run in current environment
    - select(names, environment): Get specific candidates by name
    """

    def __init__(self) -> None:
        self._registry: dict[str, Any] = {}  # name → instance or factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> BenchmarkCandidate:
        if name not in self._registry:
            raise KeyError(
                f"Unknown candidate: {name!r}\n"
                f"Available: {list(self._registry.keys())}"
            )
        entry = self._registry[name]
        if callable(entry) and not hasattr(entry, "run"):
            # It's a factory function; call it to get the instance
            instance = entry()
            self._registry[name] = instance
            return instance
        return entry

    def register(self, name: str, factory_or_instance: Any) -> None:
        self._registry[name] = factory_or_instance

    def names(self) -> list[str]:
        return list(self._registry.keys())

    def declared(self) -> list[str]:
        """Return names of all declared candidates (no dependency checks).
        
        Fix #8: This is for portable tests that don't require MLX or other
        runtime dependencies to be installed.
        """
        return self.names()

    def available(self, environment: str = "auto") -> list[str]:
        """Return names of candidates that report is_available() == True.
        
        Fix #8: Checks runtime dependencies and only returns candidates that
        can actually run in the current environment.
        
        Args:
            environment: "auto" (default), "mlx", "numpy", or "portable"
        """
        available = []
        for name in self.names():
            try:
                c = self.get(name)
                if hasattr(c, "is_available") and c.is_available():
                    # Additional environment-specific checks
                    if environment == "portable":
                        # Portable candidates don't require MLX
                        available.append(name)
                    elif environment == "mlx":
                        # MLX candidates require MLX
                        available.append(name)
                    elif environment == "numpy":
                        # Numpy candidates don't require MLX
                        available.append(name)
                    else:  # auto
                        available.append(name)
            except Exception:
                pass
        return available

    def select(self, names: list[str], environment: str = "auto") -> list[BenchmarkCandidate]:
        """Get specific candidates by name.
        
        Fix #8: This is the authoritative way to get candidates for benchmarking.
        
        Args:
            names: List of candidate names to select
            environment: "auto", "mlx", "numpy", or "portable"
        
        Returns:
            List of candidate instances
        """
        candidates = []
        for name in names:
            try:
                c = self.get(name)
                # Check availability if not auto
                if environment != "auto":
                    if hasattr(c, "is_available") and not c.is_available():
                        continue
                # Environment-specific filtering
                if environment == "portable" and not hasattr(c, "is_available"):
                    continue
                if environment == "mlx" and not hasattr(c, "is_available"):
                    continue
                if environment == "numpy" and not hasattr(c, "is_available"):
                    continue
                candidates.append(c)
            except Exception:
                pass
        return candidates

    def declared_candidates(self) -> list[str]:
        """Return names of all declared candidates (no dependency checks).
        
        Fix #8: Deprecated. Use declared() instead.
        """
        return self.declared()

    def available_candidates(self) -> list[str]:
        """Return names of candidates that report is_available() == True.
        
        Fix #8: Deprecated. Use available() instead.
        """
        return self.available()


def _make_dense_baseline() -> Any:
    from benchmarks.candidates.dense_baseline import DenseMlxBaseline
    return DenseMlxBaseline()


def _make_mlx_lm_8bit_kv() -> Any:
    """MLX-LM built-in 8-bit KV cache (primary control for comparison)."""
    from rfsn_v11.candidates.mlx_lm_quantized import MLXLMQuantizedKV
    return MLXLMQuantizedKV(kv_bits=8)


def _make_a1() -> Any:
    from benchmarks.candidates.a1_wht_grouped_k8v4_gs64 import A1_WHT_Grouped
    return A1_WHT_Grouped()


def _make_a1b() -> Any:
    from benchmarks.candidates.a1b_wht_asym_k8v4 import A1b_WHT_Asym
    return A1b_WHT_Asym()


def _make_a2() -> Any:
    from benchmarks.candidates.a2_wht_polar_4bit import A2_WHT_Polar
    return A2_WHT_Polar()


def _make_a3() -> Any:
    from benchmarks.candidates.a3_wht_turboquant_mse_4bit import (
        A3_WHT_TurboQuant_MSE,
    )
    return A3_WHT_TurboQuant_MSE()


def _make_a4() -> Any:
    from benchmarks.candidates.a4_wht_turboquant_qjl_4bit import (
        A4_WHT_TurboQuant_QJL,
    )
    return A4_WHT_TurboQuant_QJL()


def _make_b1() -> Any:
    from benchmarks.candidates.b1_sparsejl_grouped_k8v4_gs64 import (
        B1_SparseJL_Grouped,
    )
    return B1_SparseJL_Grouped()


def _make_r1() -> Any:
    from benchmarks.candidates.r1_wht_grouped_residual128 import (
        R1_WHT_Grouped_Residual,
    )
    return R1_WHT_Grouped_Residual()


def _make_r2() -> Any:
    from benchmarks.candidates.r2_turboquant_mse_residual128 import (
        R2_TurboQuant_MSE_Residual,
    )
    return R2_TurboQuant_MSE_Residual()


def _make_s1() -> Any:
    from benchmarks.candidates.s1_snapkv_prune_only import S1_SnapKV_PruneOnly
    return S1_SnapKV_PruneOnly()


def _make_s2() -> Any:
    from benchmarks.candidates.s2_snapkv_plus_grouped import (
        S2_SnapKV_PlusGrouped,
    )
    return S2_SnapKV_PlusGrouped()


def _make_s3() -> Any:
    from benchmarks.candidates.s3_snapkv_plus_turboquant_mse_residual128 import (  # noqa: E501
        S3_SnapKV_PlusTurboQuantMSEResidual,
    )
    return S3_SnapKV_PlusTurboQuantMSEResidual()


def _make_rfsn_direct_packed_k8v8() -> Any:
    """Fix #8: RFSN direct-packed K8/V8 candidate (canonical BS64)."""
    from rfsn_v11.candidates.rfsn_direct_packed_adapter import RFSNDirectPackedCandidate
    return RFSNDirectPackedCandidate(key_bits=8, value_bits=8, group_size=64, staging_capacity=64)


def _make_rfsn_direct_packed_k8v8_smoke() -> Any:
    """Fix #8: RFSN direct-packed K8/V8 smoke candidate (BS8 for fast tests)."""
    from rfsn_v11.candidates.rfsn_direct_packed_adapter import RFSNDirectPackedCandidate
    return RFSNDirectPackedCandidate(key_bits=8, value_bits=8, group_size=64, staging_capacity=8)


def _make_rfsn_direct_packed_k8v5() -> Any:
    """Fix #8: RFSN direct-packed K8/V5 candidate."""
    from rfsn_v11.candidates.rfsn_direct_packed_adapter import RFSNDirectPackedCandidate
    return RFSNDirectPackedCandidate(key_bits=8, value_bits=5, group_size=64)


def _make_rfsn_direct_packed_k8v6() -> Any:
    """Fix #8: RFSN direct-packed K8/V6 candidate."""
    from rfsn_v11.candidates.rfsn_direct_packed_adapter import RFSNDirectPackedCandidate
    return RFSNDirectPackedCandidate(key_bits=8, value_bits=6, group_size=64)


def _make_rfsn_direct_packed_k16v8() -> Any:
    """Fix #8: RFSN direct-packed K16/V8 candidate (diagnostic)."""
    from rfsn_v11.candidates.rfsn_direct_packed_adapter import RFSNDirectPackedCandidate
    return RFSNDirectPackedCandidate(key_bits=16, value_bits=8, group_size=64)


def _make_rfsn_direct_packed_k8v16() -> Any:
    """Fix #8: RFSN direct-packed K8/V16 candidate (diagnostic)."""
    from rfsn_v11.candidates.rfsn_direct_packed_adapter import RFSNDirectPackedCandidate
    return RFSNDirectPackedCandidate(key_bits=8, value_bits=16, group_size=64)


def _make_rfsn_direct_packed_k16v16() -> Any:
    """Fix #8: RFSN direct-packed K16/V16 candidate (diagnostic)."""
    from rfsn_v11.candidates.rfsn_direct_packed_adapter import RFSNDirectPackedCandidate
    return RFSNDirectPackedCandidate(key_bits=16, value_bits=16, group_size=64)


def _make_rfsn_v10_k8v5() -> Any:
    """Fix #8: RFSN v10 K8/V5 dense reconstruction candidate."""
    from rfsn_v11.candidates.rfsn_v10_adapter import RFSNV10Candidate
    return RFSNV10Candidate(key_bits=8, value_bits=5, group_size=64)


def _make_rfsn_v10_k8v8() -> Any:
    """Fix #8: RFSN v10 K8/V8 dense reconstruction candidate."""
    from rfsn_v11.candidates.rfsn_v10_adapter import RFSNV10Candidate
    return RFSNV10Candidate(key_bits=8, value_bits=8, group_size=64)


# ---------------------------------------------------------------------------
# Registry builders
# ---------------------------------------------------------------------------

# Phase 0 canonical candidates — the only ones active for validation
_CANONICAL_CANDIDATES = [
    "dense_mlx_baseline",
    "mlx_lm_8bit_kv",
    "rfsn_direct_packed_k8v8",
]


def build_default_registry() -> CandidateRegistry:
    """Build and return the FROZEN default registry.

    Phase 0 Scope Freeze: Contains exactly three canonical candidates:
      1. dense_mlx_baseline
      2. mlx_lm_8bit_kv
      3. rfsn_direct_packed_k8v8

    All other candidates are accessible via build_experimental_registry().
    """
    reg = CandidateRegistry()
    reg.register("dense_mlx_baseline", _make_dense_baseline)
    reg.register("mlx_lm_8bit_kv", _make_mlx_lm_8bit_kv)
    reg.register("rfsn_direct_packed_k8v8", _make_rfsn_direct_packed_k8v8)
    return reg


def build_experimental_registry() -> CandidateRegistry:
    """Build and return the EXPERIMENTAL registry with all known candidates.

    Includes legacy research candidates (Polar, TurboQuant, QJL, SnapKV,
    SparseJL, additional bit-widths, etc.) that are excluded from the
    default registry.  Use this only for explicit research work.
    """
    reg = CandidateRegistry()
    # Baselines and controls
    reg.register("dense_mlx_baseline", _make_dense_baseline)
    reg.register("mlx_lm_8bit_kv", _make_mlx_lm_8bit_kv)
    # Phase 3-10 legacy candidates (archived, not actively benchmarked)
    reg.register("A1_wht_grouped_k8v4_gs64", _make_a1)
    reg.register("A1b_wht_asym_k8v4", _make_a1b)
    reg.register("A2_wht_polar_4bit", _make_a2)
    reg.register("A3_wht_turboquant_mse_4bit", _make_a3)
    reg.register("A4_wht_turboquant_qjl_4bit", _make_a4)
    reg.register("B1_sparsejl_grouped_k8v4_gs64", _make_b1)
    reg.register("R1_wht_grouped_residual128", _make_r1)
    reg.register("R2_turboquant_mse_residual128", _make_r2)
    reg.register("S1_snapkv_prune_only", _make_s1)
    reg.register("S2_snapkv_plus_grouped", _make_s2)
    reg.register("S3_snapkv_plus_turboquant_mse_residual128", _make_s3)
    # RFSN direct-packed variants (all bit-widths)
    reg.register("rfsn_direct_packed_k8v8", _make_rfsn_direct_packed_k8v8)
    reg.register("rfsn_direct_packed_k8v8_smoke", _make_rfsn_direct_packed_k8v8_smoke)
    reg.register("rfsn_direct_packed_k8v5", _make_rfsn_direct_packed_k8v5)
    reg.register("rfsn_direct_packed_k8v6", _make_rfsn_direct_packed_k8v6)
    reg.register("rfsn_direct_packed_k16v8", _make_rfsn_direct_packed_k16v8)
    reg.register("rfsn_direct_packed_k8v16", _make_rfsn_direct_packed_k8v16)
    reg.register("rfsn_direct_packed_k16v16", _make_rfsn_direct_packed_k16v16)
    # RFSN v10 dense-reconstruction candidates
    reg.register("rfsn_v10_k8v5", _make_rfsn_v10_k8v5)
    reg.register("rfsn_v10_k8v8", _make_rfsn_v10_k8v8)
    return reg


# Module-level default
_default_registry: CandidateRegistry | None = None
_experimental_registry: CandidateRegistry | None = None


def get_registry() -> CandidateRegistry:
    """Return the module-level default (frozen) registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = build_default_registry()
    return _default_registry


def get_experimental_registry() -> CandidateRegistry:
    """Return the module-level experimental registry."""
    global _experimental_registry
    if _experimental_registry is None:
        _experimental_registry = build_experimental_registry()
    return _experimental_registry
