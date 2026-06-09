#!/usr/bin/env python3
"""Model download utility for RFSN v10 production validation.

Downloads open-source models from Hugging Face for validation purposes.
Supports Llama, Mistral, and other compatible causal LMs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: huggingface_hub and transformers required for model download")
    print("Install with: pip install huggingface_hub transformers")
    sys.exit(1)


# Recommended models for production validation
RECOMMENDED_MODELS = {
    "llama-3-8b": {
        "repo_id": "meta-llama/Meta-Llama-3-8B",
        "requires_auth": True,
        "description": "Llama 3 8B - Production-grade model",
    },
    "mistral-7b": {
        "repo_id": "mistralai/Mistral-7B-v0.3",
        "requires_auth": False,
        "description": "Mistral 7B v0.3 - Open source, no auth required",
    },
    "phi-3-mini": {
        "repo_id": "microsoft/Phi-3-mini-4k-instruct",
        "requires_auth": False,
        "description": "Phi-3 Mini - Small, efficient model",
    },
    "qwen-7b": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "requires_auth": False,
        "description": "Qwen 2.5 7B - Strong multilingual model",
    },
}


def download_model(
    repo_id: str,
    output_dir: Path,
    token: str | None = None,
    resume_download: bool = True,
) -> Path:
    """Download a model from Hugging Face.

    Args:
        repo_id: Hugging Face repository ID (e.g., "mistralai/Mistral-7B-v0.3")
        output_dir: Directory to download model to
        token: Optional Hugging Face auth token (required for some models)
        resume_download: Whether to resume interrupted downloads

    Returns:
        Path to downloaded model directory
    """
    print(f"Downloading model: {repo_id}")
    print(f"Output directory: {output_dir}")

    try:
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            resume_download=resume_download,
            token=token,
        )
        print(f"Model downloaded successfully to: {model_path}")

        # Verify tokenizer can be loaded
        print("Verifying tokenizer...")
        AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print("Tokenizer verified successfully")

        return Path(model_path)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"ERROR: Failed to download model: {e}")
        if "requires_auth" in str(e).lower() or "gated" in str(e).lower():
            print(f"Model {repo_id} requires authentication.")
            print("Get a token from: https://huggingface.co/settings/tokens")
            print("Then run with: --token YOUR_TOKEN")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models for RFSN validation")
    parser.add_argument(
        "model",
        nargs="?",
        choices=list(RECOMMENDED_MODELS.keys()) + ["custom"],
        help="Model to download (or 'custom' for custom repo_id)",
    )
    parser.add_argument("--repo-id", help="Custom Hugging Face repo ID (required if model=custom)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for downloaded models",
    )
    parser.add_argument("--token", help="Hugging Face auth token (required for gated models)")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume interrupted downloads",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recommended models and exit",
    )

    args = parser.parse_args()

    if args.list:
        print("Recommended models for validation:")
        for key, info in RECOMMENDED_MODELS.items():
            auth_status = "[AUTH REQUIRED]" if info["requires_auth"] else "[OPEN]"
            print(f"  {key:20s} {auth_status:20s} {info['description']}")
            print(f"  {'':20s} Repo: {info['repo_id']}")
        return

    if not args.model:
        parser.error("Please specify a model or use --list to see options")

    if args.model == "custom":
        if not args.repo_id:
            parser.error("--repo-id required when model=custom")
        repo_id = args.repo_id
    else:
        repo_id = RECOMMENDED_MODELS[args.model]["repo_id"]
        if RECOMMENDED_MODELS[args.model]["requires_auth"] and not args.token:
            print(f"WARNING: Model {args.model} requires authentication")
            print("Get a token from: https://huggingface.co/settings/tokens")
            print("Then run with: --token YOUR_TOKEN")
            print()
            response = input("Continue anyway? (y/N): ")
            if response.lower() != "y":
                sys.exit(0)

    output_dir = args.output_dir / args.model.replace("_", "-")
    download_model(
        repo_id=repo_id,
        output_dir=output_dir,
        token=args.token,
        resume_download=not args.no_resume,
    )


if __name__ == "__main__":
    main()
