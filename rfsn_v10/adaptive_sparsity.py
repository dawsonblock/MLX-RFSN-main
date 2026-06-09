"""RFSN v10 Adaptive Sparsity Controller with split sparse/quant metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveDecision:
    top_k_ratio: float
    disable_sparse: bool
    disable_quantized: bool
    reason: str


@dataclass
class _ProfileState:
    top_k_ratio: float
    stable_clean_steps: int = 0
    disable_sparse: bool = False
    disable_quantized: bool = False
    reason: str = "initial"


class AdaptiveSparsityController:
    """Adjust sparse and quantized behavior per profile key."""

    def __init__(
        self,
        initial_top_k_ratio: float = 1.0,
        min_top_k_ratio: float = 0.125,
        max_top_k_ratio: float = 1.0,
        sparse_cosine_warn: float = 0.95,
        sparse_cosine_fail: float = 0.90,
        sparse_mae_warn: float = 0.05,
        quant_cosine_warn: float = 0.97,
        quant_cosine_fail: float = 0.94,
        quant_mae_warn: float = 0.05,
        decrease_step_size: float = 0.05,
        increase_step_size: float = 0.15,
        stabilization_steps: int = 3,
    ):
        self.initial_top_k_ratio = float(initial_top_k_ratio)
        self.min_top_k_ratio = float(min_top_k_ratio)
        self.max_top_k_ratio = float(max_top_k_ratio)
        self.sparse_cosine_warn = float(sparse_cosine_warn)
        self.sparse_cosine_fail = float(sparse_cosine_fail)
        self.sparse_mae_warn = float(sparse_mae_warn)
        self.quant_cosine_warn = float(quant_cosine_warn)
        self.quant_cosine_fail = float(quant_cosine_fail)
        self.quant_mae_warn = float(quant_mae_warn)
        self.decrease_step_size = float(decrease_step_size)
        self.increase_step_size = float(increase_step_size)
        self.stabilization_steps = int(stabilization_steps)
        self._profiles: dict[str, _ProfileState] = {}

    @staticmethod
    def _seq_bucket(seq_len: int | None) -> str:
        if seq_len is None:
            return "unknown"
        val = int(seq_len)
        if val <= 256:
            return "s"
        if val <= 1024:
            return "m"
        if val <= 4096:
            return "l"
        return "xl"

    def _profile_key(
        self,
        *,
        model_id: str | None,
        layer_id: str | None,
        skill_pattern: str | None,
        seq_len: int | None,
    ) -> str:
        return ":".join(
            [
                model_id or "default_model",
                layer_id or "default_layer",
                self._seq_bucket(seq_len),
                skill_pattern or "default_skill",
            ]
        )

    def _state_for(
        self,
        *,
        model_id: str | None,
        layer_id: str | None,
        skill_pattern: str | None,
        seq_len: int | None,
    ) -> _ProfileState:
        key = self._profile_key(
            model_id=model_id,
            layer_id=layer_id,
            skill_pattern=skill_pattern,
            seq_len=seq_len,
        )
        state = self._profiles.get(key)
        if state is None:
            state = _ProfileState(top_k_ratio=self.initial_top_k_ratio)
            self._profiles[key] = state
        return state

    @staticmethod
    def _bad_cosine(value: float | None, threshold: float) -> bool:
        return value is not None and float(value) < float(threshold)

    @staticmethod
    def _bad_mae(value: float | None, threshold: float) -> bool:
        return value is not None and float(value) > float(threshold)

    def get_decision(
        self,
        *,
        model_id: str | None = None,
        layer_id: str | None = None,
        skill_pattern: str | None = None,
        seq_len: int | None = None,
    ) -> AdaptiveDecision:
        state = self._state_for(
            model_id=model_id,
            layer_id=layer_id,
            skill_pattern=skill_pattern,
            seq_len=seq_len,
        )
        return AdaptiveDecision(
            top_k_ratio=state.top_k_ratio,
            disable_sparse=state.disable_sparse,
            disable_quantized=state.disable_quantized,
            reason=state.reason,
        )

    def get_top_k_ratio(self) -> float:
        """Backward-compatible global/default top-k ratio getter."""
        return self.get_decision().top_k_ratio

    def update(
        self,
        *,
        sparse_success: bool,
        fallback_used: bool,
        sparse_audit_cosine: float | None = None,
        sparse_audit_rel_mae: float | None = None,
        quant_audit_cosine: float | None = None,
        quant_audit_rel_mae: float | None = None,
        model_id: str | None = None,
        layer_id: str | None = None,
        skill_pattern: str | None = None,
        seq_len: int | None = None,
    ) -> AdaptiveDecision:
        state = self._state_for(
            model_id=model_id,
            layer_id=layer_id,
            skill_pattern=skill_pattern,
            seq_len=seq_len,
        )

        sparse_bad = (
            self._bad_cosine(sparse_audit_cosine, self.sparse_cosine_warn)
            or self._bad_mae(sparse_audit_rel_mae, self.sparse_mae_warn)
        )
        sparse_fail = self._bad_cosine(sparse_audit_cosine, self.sparse_cosine_fail)

        quant_bad = (
            self._bad_cosine(quant_audit_cosine, self.quant_cosine_warn)
            or self._bad_mae(quant_audit_rel_mae, self.quant_mae_warn)
        )
        quant_fail = self._bad_cosine(quant_audit_cosine, self.quant_cosine_fail)

        if not sparse_success or fallback_used:
            state.top_k_ratio = min(
                self.max_top_k_ratio,
                state.top_k_ratio + self.increase_step_size,
            )
            state.disable_sparse = True
            state.reason = "fallback_or_sparse_failure"
            state.stable_clean_steps = 0
        elif sparse_bad:
            state.top_k_ratio = min(
                self.max_top_k_ratio,
                state.top_k_ratio + self.increase_step_size,
            )
            state.disable_sparse = sparse_fail
            state.reason = "sparse_quality_degraded"
            state.stable_clean_steps = 0
        elif quant_bad:
            state.disable_quantized = quant_fail
            state.reason = "quant_quality_degraded"
            state.stable_clean_steps = 0
        else:
            state.stable_clean_steps += 1
            if state.stable_clean_steps >= self.stabilization_steps:
                state.top_k_ratio = max(
                    self.min_top_k_ratio,
                    state.top_k_ratio - self.decrease_step_size,
                )
                state.disable_sparse = False
                state.disable_quantized = False
                state.reason = "stable_clean_reduce_topk"
                state.stable_clean_steps = 0
            else:
                state.reason = "stable_clean_hold"

        return AdaptiveDecision(
            top_k_ratio=state.top_k_ratio,
            disable_sparse=state.disable_sparse,
            disable_quantized=state.disable_quantized,
            reason=state.reason,
        )

    def reset(self) -> None:
        self._profiles.clear()

    def get_stats(self) -> dict:
        if not self._profiles:
            return {"profiles": 0}

        topks = [state.top_k_ratio for state in self._profiles.values()]
        sparse_disabled = sum(1 for state in self._profiles.values() if state.disable_sparse)
        quant_disabled = sum(
            1 for state in self._profiles.values() if state.disable_quantized
        )
        return {
            "profiles": len(self._profiles),
            "min_top_k_ratio": min(topks),
            "max_top_k_ratio": max(topks),
            "sparse_disabled_profiles": sparse_disabled,
            "quant_disabled_profiles": quant_disabled,
        }
