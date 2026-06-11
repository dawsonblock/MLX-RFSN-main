"""KV-compression candidate interface for the shootout benchmark.

All candidates implement KVCompressionCandidate and return CandidateResult.
Quality evaluation uses the thresholds in quality_gates.py.

Usage
-----
from rfsn_v11.candidates.base import KVCompressionCandidate, CandidateResult
from rfsn_v11.candidates.mlx_lm_baseline import MLXLMBaseline
from rfsn_v11.candidates.rfsn_v10_adapter import RFSNV10Candidate
from rfsn_v11.candidates.rfsn_v11_adapter import RFSNV11Candidate
from rfsn_v11.candidates.turboquant_v2_adapter import TurboQuantV2Candidate
from rfsn_v11.candidates.polar_reference_adapter import PolarReferenceAdapter
from rfsn_v11.candidates.turbo_polar_adapter import TurboPolarAdapter
from rfsn_v11.candidates.turbo_polar_config import TurboPolarConfig
from rfsn_v11.candidates.quality_gates import evaluate_quality_gate
"""
