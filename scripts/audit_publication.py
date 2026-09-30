#!/usr/bin/env python3
"""Audit the exact GitHub and Hugging Face publication payloads."""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAX_GITHUB_FILE_BYTES = 20 * 1024 * 1024

ROOT_FILES = {
    ".gitignore",
    "README.md",
    "REPORT.md",
    "PUBLISHING.md",
    "LICENSE",
    "CITATION.cff",
    "THIRD_PARTY_DATA.md",
    "requirements.txt",
}
PUBLIC_DIRS = {"assets", "configs", "results", "scripts", "src", "tests", "publish"}
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".txt", ".cff", ".svg"}

SECRET_PATTERNS = {
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "password assignment": re.compile(
        r"(?i)\b(?:password|passwd|pwd)\s*[:=]\s*[^\s<>{}]{6,}"
    ),
}


def publication_files() -> list[Path]:
    files = [PROJECT_ROOT / name for name in sorted(ROOT_FILES)]
    for dirname in sorted(PUBLIC_DIRS):
        root = PROJECT_ROOT / dirname
        if root.exists():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.name != ".DS_Store"
            )
    return sorted(set(files))


def main() -> None:
    errors: list[str] = []
    files = publication_files()
    local_path_marker = "/" + "Users/"
    windows_path_marker = "\\" + "Users\\"

    for path in files:
        relative = path.relative_to(PROJECT_ROOT)
        if path.is_symlink():
            errors.append(f"symlink is not allowed: {relative}")
            continue
        if path.stat().st_size > MAX_GITHUB_FILE_BYTES:
            errors.append(
                f"oversized GitHub file ({path.stat().st_size} bytes): {relative}"
            )
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".gitignore",
            "LICENSE",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name != "audit_publication.py":
            if local_path_marker in text or windows_path_marker in text:
                errors.append(f"machine-local user path: {relative}")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"possible {label}: {relative}")

    blocked_roots = ["data", "outputs", "adapters", ".venv"]
    tracked_roots = {path.parts[0] for path in (p.relative_to(PROJECT_ROOT) for p in files)}
    overlap = sorted(set(blocked_roots) & tracked_roots)
    if overlap:
        errors.append(f"blocked generated directories entered payload: {overlap}")

    if errors:
        print("Publication audit failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print(f"Publication audit passed for {len(files)} GitHub files.")
    print("No likely credentials, private machine paths, or oversized files found.")
    print("Generated data, outputs, adapters, and environments are excluded from GitHub.")


if __name__ == "__main__":
    main()
