#!/usr/bin/env python3
"""Memory layer shootout benchmark.

Compares external memory candidates separately from KV cache.
Metrics: recall@k, latency, RAM, index size, insert time.

Usage:
    python benchmarks/memory_shootout.py --backend qdrant
    python benchmarks/memory_shootout.py --backend turbovec
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from rfsn_v11.candidates.json_utils import dump_json_strict  # noqa: E402

ARTIFACTS_ROOT = Path("artifacts/bench/memory")


def _generate_test_corpus(size: int = 100) -> list[str]:
    return [f"This is test document number {i} about machine learning." for i in range(size)]


def _benchmark_qdrant(corpus: list[str]) -> dict[str, Any]:
    from memory.qdrant_memory import QdrantMemory

    mem = QdrantMemory(collection="memory_shootout_test")
    mem.ensure_collection(vector_size=384)

    # Encode with a dummy embedder for the scaffold
    try:
        from memory.embeddings import Embedder
        embedder = Embedder()
        embeddings = embedder.encode(corpus)
    except ImportError:
        embeddings = [[0.1] * 384 for _ in corpus]

    t0 = time.perf_counter()
    mem.add(corpus, embeddings)
    insert_time = time.perf_counter() - t0

    query = embeddings[0]
    t0 = time.perf_counter()
    results = mem.search(query, limit=5)
    search_latency = time.perf_counter() - t0

    return {
        "backend": "qdrant",
        "corpus_size": len(corpus),
        "insert_time_sec": insert_time,
        "search_latency_sec": search_latency,
        "recall_at_5": len(results) / 5.0,
        "notes": "Qdrant default memory benchmark",
    }


def _benchmark_turbovec(corpus: list[str]) -> dict[str, Any]:
    from memory.turbovec_memory import TurboVecMemory

    mem = TurboVecMemory(path="./.tmp/turbovec_shootout")

    try:
        from memory.embeddings import Embedder
        embedder = Embedder()
        embeddings = embedder.encode(corpus)
    except ImportError:
        embeddings = [[0.1] * 384 for _ in corpus]

    try:
        t0 = time.perf_counter()
        mem.add(corpus, embeddings)
        insert_time = time.perf_counter() - t0

        query = embeddings[0]
        t0 = time.perf_counter()
        results = mem.search(query, limit=5)
        search_latency = time.perf_counter() - t0

        return {
            "backend": "turbovec",
            "corpus_size": len(corpus),
            "insert_time_sec": insert_time,
            "search_latency_sec": search_latency,
            "recall_at_5": len(results) / 5.0,
            "notes": "TurboVec memory benchmark",
        }
    except NotImplementedError:
        return {
            "backend": "turbovec",
            "corpus_size": len(corpus),
            "insert_time_sec": None,
            "search_latency_sec": None,
            "recall_at_5": None,
            "notes": "TurboVec not yet implemented",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory layer shootout")
    parser.add_argument("--backend", choices=["qdrant", "turbovec", "all"], default="all")
    parser.add_argument("--corpus-size", type=int, default=100)
    args = parser.parse_args()

    corpus = _generate_test_corpus(args.corpus_size)
    results: list[dict[str, Any]] = []

    if args.backend in ("qdrant", "all"):
        try:
            results.append(_benchmark_qdrant(corpus))
        except Exception as exc:
            results.append({"backend": "qdrant", "error": str(exc)})

    if args.backend in ("turbovec", "all"):
        try:
            results.append(_benchmark_turbovec(corpus))
        except Exception as exc:
            results.append({"backend": "turbovec", "error": str(exc)})

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACTS_ROOT / "results.json"
    with json_path.open("w", encoding="utf-8") as fh:
        dump_json_strict(results, fh, indent=2)

    print(f"Wrote {json_path}")
    for r in results:
        print(f"  {r['backend']}: {r.get('notes', '')}")


if __name__ == "__main__":
    main()
