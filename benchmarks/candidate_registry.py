"""RFSN candidate registry.

Maps canonical candidate names to instantiated candidate objects.
All imports are lazy — missing optional dependencies (mlx,
    external/turboquant-mlx, external/mlx-turboquant) only raise
    when the specific candidate is requested.

Canonical name conventions
---------------------------
A1_wht_grouped_k8v4_gs64  — Phase 3: grouped WHT, keys 8-bit,
    values 4-bit, group 64
A1b_wht_asym_k8v4            — Phase 4: asymmetric bit sweep
A2_wht_polar_4bit            — Phase 6: PolarQuant
A3_wht_turboquant_mse_4bit   — Phase 5: TurboQuant MSE-only
A4_wht_turboquant_qjl_4bit   — Phase 10: TurboQuant + QJL
B1_sparsejl_grouped_k8v4_gs64 — Phase 9:
    Sparse JL ablation
R1_wht_grouped_residual128 — Phase 7: grouped WHT +
    FP16 residual window R=128
R2_turboquant_mse_residual128 — Phase 7:
    TurboQuant MSE + FP16 residual window R=128
S1_snapkv_prune_only         — Phase 8: SnapKV pruning only
S2_snapkv_plus_grouped       — Phase 8: SnapKV + grouped WHT
S3_snapkv_plus_turboquant_mse_residual128 — Phase 8:
    SnapKV + TurboQuant MSE + residual
dense_mlx_baseline           — Phase 1: dense FP16 baseline (always available)
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
    """Lazy registry of all benchmark candidates.

    get(name) instantiates the candidate on first call.
    list_available() returns names of candidates whose
    dependencies are satisfied.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Any] = {}  # name → instance or factory

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, name: str) -> "BenchmarkCandidate":
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

    def list_available(self) -> list[str]:
        """Return names of candidates that report is_available() == True."""
        available = []
        for name in self.names():
            try:
                c = self.get(name)
                if hasattr(c, "is_available") and c.is_available():
                    available.append(name)
            except Exception:
                pass
        return available


def _make_dense_baseline() -> Any:
    from benchmarks.candidates.dense_baseline import DenseMlxBaseline
    return DenseMlxBaseline()


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


# ---------------------------------------------------------------------------
# Default registry instance
# ---------------------------------------------------------------------------

def build_default_registry() -> CandidateRegistry:
    """Build and return the standard registry with all known candidates."""
    reg = CandidateRegistry()
    reg.register("dense_mlx_baseline", _make_dense_baseline)
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
    return reg


# Module-level default
_default_registry: CandidateRegistry | None = None


def get_registry() -> CandidateRegistry:
    """Return the module-level default registry (created on first call)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = build_default_registry()
    return _default_registry
