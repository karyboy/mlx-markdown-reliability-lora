#!/usr/bin/env python3
"""Publish the allowlisted final adapter artifacts to Hugging Face Hub."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "logicless/qwen25-coder-3b-markdown-reliability-lora-mlx"
EXPECTED_ADAPTER_SHA256 = (
    "a892954dbf11822d6f4ab1ef50328b6ade8ca3fad072b8706eb319dfaa27e5d5"
)

UPLOADS = {
    "publish/huggingface/README.md": "README.md",
    "LICENSE": "LICENSE",
    "adapters/qwen25-coder-3b-markdown-reliability-wrapper-run2/adapters.safetensors": "adapters.safetensors",
    "adapters/qwen25-coder-3b-markdown-reliability-wrapper-run2/adapter_config.json": "adapter_config.json",
    "configs/experiment.yaml": "configs/experiment.yaml",
    "configs/lora.yaml": "configs/lora.yaml",
    "configs/lora_wrapper_booster.yaml": "configs/lora_wrapper_booster.yaml",
    "results/final_comparison.json": "results/final_comparison.json",
    "results/wrapper_run2_latentmd_metrics.json": "results/latentmd_metrics.json",
    "results/wrapper_run2_training_summary.json": "results/training_summary.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_payload() -> list[tuple[Path, str]]:
    payload: list[tuple[Path, str]] = []
    for source, destination in UPLOADS.items():
        path = PROJECT_ROOT / source
        if not path.is_file():
            raise SystemExit(f"Required publication file is missing: {path}")
        payload.append((path, destination))

    adapter_path = PROJECT_ROOT / next(
        source for source in UPLOADS if source.endswith("/adapters.safetensors")
    )
    actual_hash = sha256(adapter_path)
    if actual_hash != EXPECTED_ADAPTER_SHA256:
        raise SystemExit(
            "Final adapter SHA-256 mismatch. Refusing upload. "
            f"Expected {EXPECTED_ADAPTER_SHA256}, got {actual_hash}."
        )

    config_path = PROJECT_ROOT / next(
        source for source in UPLOADS if source.endswith("/adapter_config.json")
    )
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    serialized = json.dumps(adapter_config)
    local_path_marker = "/" + "Users/"
    windows_path_marker = "\\" + "Users\\"
    if local_path_marker in serialized or windows_path_marker in serialized:
        raise SystemExit("adapter_config.json contains a machine-local path.")
    if adapter_config.get("model") != "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit":
        raise SystemExit("adapter_config.json has an unexpected base model.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list the payload without creating or uploading.",
    )
    args = parser.parse_args()
    payload = validate_payload()

    print(f"Validated {len(payload)} allowlisted files for {args.repo_id}:")
    for source, destination in payload:
        print(f"  {source.relative_to(PROJECT_ROOT)} -> {destination}")
    if args.dry_run:
        return

    api = HfApi()
    identity = api.whoami()
    print(f"Authenticated to Hugging Face as {identity['name']}.")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    for source, destination in payload:
        print(f"Uploading {destination}...")
        api.upload_file(
            path_or_fileobj=source,
            path_in_repo=destination,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Publish {destination}",
        )
    print(f"Published: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
