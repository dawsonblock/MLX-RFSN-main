"""Evidence-based promotion policy for RFSN candidates.

This module replaces hardcoded promotion decisions with policy-based
validation that checks actual prerequisites before allowing promotion.
"""
from __future__ import annotations

from typing import Any


class PromotionPolicy:
    """Evidence-based promotion policy.

    Promotion is only allowed when all prerequisites are satisfied:
    - Strict current-run token provenance
    - Runtime trace validation
    - Real cache injection
    - Actual memory measurements
    - Zero benchmark errors
    - Candidate status eligible for promotion
    - Multiple required models and contexts
    - Clean source tree
    - Matching source and artifact release IDs
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize promotion policy with optional configuration.

        Args:
            config: Policy configuration dictionary with optional overrides.
        """
        self.config = config or {}

    def all_prerequisites_satisfied(self, run_bundle: dict[str, Any]) -> bool:
        """Check if all promotion prerequisites are satisfied.

        Args:
            run_bundle: Dictionary containing run metadata and results.

        Returns:
            True if all prerequisites are satisfied, False otherwise.
        """
        prerequisites = [
            self._check_token_provenance(run_bundle),
            self._check_runtime_trace_validation(run_bundle),
            self._check_real_cache_injection(run_bundle),
            self._check_actual_memory_measurements(run_bundle),
            self._check_zero_benchmark_errors(run_bundle),
            self._check_candidate_eligibility(run_bundle),
            self._check_multi_model_coverage(run_bundle),
            self._check_clean_source_tree(run_bundle),
            self._check_release_id_match(run_bundle),
        ]

        return all(prerequisites)

    def _check_token_provenance(self, run_bundle: dict[str, Any]) -> bool:
        """Check that token sequence has strict provenance."""
        metadata = run_bundle.get("metadata", {})

        # Must have non-empty token sequence hash
        token_hash = metadata.get("token_sequence_hash", "")
        if not token_hash:
            return False

        # Must reference source artifact (not inherited)
        provenance = metadata.get("token_sequence_provenance")
        if not provenance:
            return False

        # Must have artifact reference with SHA256
        if "token_sequence_artifact" not in provenance:
            return False
        if "token_sequence_artifact_sha256" not in provenance:
            return False

        return True

    def _check_runtime_trace_validation(self, run_bundle: dict[str, Any]) -> bool:
        """Check that runtime traces are validated."""
        results = run_bundle.get("results", [])

        for result in results:
            # Must have runtime counters
            if not result.get("packed_attention_calls"):
                return False

            # Must have zero dense fallback in strict mode
            if result.get("dense_fallback_calls", 0) > 0:
                return False

            # Must have execution backend recorded
            if not result.get("execution_backend"):
                return False

        return True

    def _check_real_cache_injection(self, run_bundle: dict[str, Any]) -> bool:
        """Check that real cache injection occurred."""
        results = run_bundle.get("results", [])

        for result in results:
            # Must have cache backend used
            if not result.get("cache_backend_used"):
                return False

            # Must not be offline-only
            if "offline" in result.get("cache_backend_used", "").lower():
                return False

            # Must have non-zero cache bytes
            if result.get("packed_bytes_written", 0) == 0:
                return False

        return True

    def _check_actual_memory_measurements(self, run_bundle: dict[str, Any]) -> bool:
        """Check that actual memory measurements are used."""
        results = run_bundle.get("results", [])

        for result in results:
            # Must have actual KV memory
            if not result.get("actual_kv_memory_mb"):
                return False

            # Must not be estimated (check for measurement_kind field)
            if result.get("measurement_kind") == "ESTIMATED":
                return False

        return True

    def _check_zero_benchmark_errors(self, run_bundle: dict[str, Any]) -> bool:
        """Check that benchmark completed without errors."""
        results = run_bundle.get("results", [])

        for result in results:
            # Must not have ERROR status
            if result.get("gate_status") == "ERROR":
                return False

            # Must not have error field populated
            if result.get("error"):
                return False

        return True

    def _check_candidate_eligibility(self, run_bundle: dict[str, Any]) -> bool:
        """Check that candidate status is eligible for promotion."""
        results = run_bundle.get("results", [])

        for result in results:
            # Must have promotion_eligible=True
            if not result.get("promotion_eligible"):
                return False

            # Must not be REFERENCE_ONLY
            if result.get("candidate_status") == "REFERENCE_ONLY":
                return False

            # Must not be CONTROL
            if result.get("candidate_status") == "CONTROL":
                return False

        return True

    def _check_multi_model_coverage(self, run_bundle: dict[str, Any]) -> bool:
        """Check that multiple models and contexts were tested."""
        metadata = run_bundle.get("metadata", {})

        # Must have tested at least 2 models (configurable)
        min_models = self.config.get("min_models", 2)
        models_tested = metadata.get("models_tested", [])
        if len(models_tested) < min_models:
            return False

        # Must have tested multiple contexts (configurable)
        min_contexts = self.config.get("min_contexts", 2)
        contexts_tested = metadata.get("contexts_tested", [])
        if len(contexts_tested) < min_contexts:
            return False

        return True

    def _check_clean_source_tree(self, run_bundle: dict[str, Any]) -> bool:
        """Check that source tree was clean when artifacts were generated."""
        metadata = run_bundle.get("metadata", {})

        # Must have git state
        git_state = metadata.get("git", {})
        if not git_state:
            return False

        # Must not be dirty
        if git_state.get("dirty", False):
            return False

        return True

    def _check_release_id_match(self, run_bundle: dict[str, Any]) -> bool:
        """Check that source and artifact release IDs match."""
        metadata = run_bundle.get("metadata", {})

        # Must have release_id in metadata
        artifact_release_id = metadata.get("release_id")
        if not artifact_release_id:
            return False

        # Must match current release_id from release.toml
        # (This would be loaded from release.toml in actual use)
        current_release_id = self.config.get("current_release_id", "alpha-8.4")
        if artifact_release_id != current_release_id:
            return False

        return True

    def get_promotion_blockers(self, run_bundle: dict[str, Any]) -> list[str]:
        """Get list of promotion blockers for debugging.

        Args:
            run_bundle: Dictionary containing run metadata and results.

        Returns:
            List of blocker descriptions.
        """
        blockers = []

        if not self._check_token_provenance(run_bundle):
            blockers.append("Token provenance missing or invalid")

        if not self._check_runtime_trace_validation(run_bundle):
            blockers.append("Runtime trace validation failed")

        if not self._check_real_cache_injection(run_bundle):
            blockers.append("Real cache injection not detected")

        if not self._check_actual_memory_measurements(run_bundle):
            blockers.append("Actual memory measurements missing")

        if not self._check_zero_benchmark_errors(run_bundle):
            blockers.append("Benchmark errors detected")

        if not self._check_candidate_eligibility(run_bundle):
            blockers.append("Candidate not eligible for promotion")

        if not self._check_multi_model_coverage(run_bundle):
            blockers.append("Insufficient model/context coverage")

        if not self._check_clean_source_tree(run_bundle):
            blockers.append("Source tree was dirty")

        if not self._check_release_id_match(run_bundle):
            blockers.append("Release ID mismatch")

        return blockers


def evaluate_promotion_eligibility(
    run_bundle: dict[str, Any],
    policy_config: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate promotion eligibility using evidence-based policy.

    Args:
        run_bundle: Dictionary containing run metadata and results.
        policy_config: Optional policy configuration overrides.

    Returns:
        Tuple of (is_eligible, blockers_list).
    """
    policy = PromotionPolicy(policy_config)
    is_eligible = policy.all_prerequisites_satisfied(run_bundle)
    blockers = policy.get_promotion_blockers(run_bundle) if not is_eligible else []

    return is_eligible, blockers
