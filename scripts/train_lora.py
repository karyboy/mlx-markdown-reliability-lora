#!/usr/bin/env python3
"""Run the pinned MLX-LM LoRA SFT after baseline and data checks pass."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable_adapter_config(
    path: Path,
    config: dict,
    config_path: Path,
    actual_iters: int | None = None,
) -> None:
    """Keep all MLX training fields while replacing machine-local paths."""
    adapter_config_path = path / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_config["model"] = config["model"]
    adapter_config["model_revision"] = config["model_revision"]
    try:
        portable_config_path = str(config_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        portable_config_path = config_path.name
    adapter_config["config"] = portable_config_path
    adapter_config["data"] = config["data"]
    adapter_config["adapter_path"] = config["adapter_path"]
    if actual_iters is not None:
        planned_iters = adapter_config.get("iters", config["iters"])
        adapter_config["planned_iters"] = planned_iters
        adapter_config["iters"] = actual_iters
        adapter_config["stopped_at_iteration"] = actual_iters
        adapter_config["early_stopped"] = actual_iters < planned_iters
    adapter_config_path.write_text(
        json.dumps(adapter_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "lora.yaml"
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow the pinned model snapshot to download if it is not cached.",
    )
    parser.add_argument(
        "--allow-without-baseline",
        action="store_true",
        help="Bypass the frozen LatentMD baseline guard.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the training command without starting MLX-LM.",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    train_path = PROJECT_ROOT / config["data"] / "train.jsonl"
    valid_path = PROJECT_ROOT / config["data"] / "valid.jsonl"
    baseline = PROJECT_ROOT / "outputs" / "latentmd_baseline_predictions_metrics.json"
    if not train_path.exists() or not valid_path.exists():
        raise SystemExit(
            "Final SFT data is missing. Run: ../.venv/bin/python "
            "scripts/generate_sft_dataset.py"
        )
    if not baseline.exists() and not args.allow_without_baseline:
        raise SystemExit(
            "Frozen baseline metrics are missing. Generate and score the 240-example "
            "LatentMD baseline first, or pass --allow-without-baseline."
        )

    executable = PROJECT_ROOT.parent / ".venv" / "bin" / "mlx_lm.lora"
    if not executable.exists():
        raise SystemExit(f"MLX-LM executable not found: {executable}")
    try:
        snapshot = snapshot_download(
            repo_id=config["model"],
            revision=config["model_revision"],
            local_files_only=not args.allow_download,
        )
    except Exception as exc:
        raise SystemExit(
            "The pinned base model is not available in the Hugging Face cache. "
            "Re-run this command with --allow-download after confirming the download. "
            f"Original error: {exc}"
        ) from exc

    command = [
        str(executable),
        "--config",
        str(args.config.resolve()),
        "--model",
        snapshot,
    ]
    if args.dry_run:
        print("Training preflight passed.")
        print("Command:", " ".join(command))
        return

    adapter_path = PROJECT_ROOT / config["adapter_path"]
    try:
        subprocess.run(
            command,
            check=True,
            cwd=PROJECT_ROOT,
        )
    except KeyboardInterrupt:
        checkpoints = sorted(adapter_path.glob("*_adapters.safetensors"))
        if checkpoints and (adapter_path / "adapter_config.json").exists():
            actual_iters = max(int(path.name.split("_", 1)[0]) for path in checkpoints)
            portable_adapter_config(
                adapter_path, config, args.config, actual_iters=actual_iters
            )
            print(f"\nPreserved and finalized iteration-{actual_iters} adapter.")
        raise
    portable_adapter_config(
        adapter_path, config, args.config, actual_iters=config["iters"]
    )
    print(f"Adapter complete: {adapter_path.resolve()}")


if __name__ == "__main__":
    main()
