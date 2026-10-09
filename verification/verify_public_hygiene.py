#!/usr/bin/env python3
"""Scan curated release text and filenames for private-ledger leakage.

This is a release hygiene check, not a guarantee against every secret format.
Source import still requires explicit privacy and redistribution review.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".md", ".txt", ".py", ".csv", ".json", ".yml", ".yaml", ".cff",
                 ".sha256", ".lock", ".toml", ".ini", ".cfg", ".log", ".rst", ".sh",
                 ".ps1", ".bat", ".html", ".xml", ".js", ".ts", ".ipynb"}
FORBIDDEN = {
    "Windows absolute path": re.compile(r"\b[A-Za-z]:[\\/]"),
    "UNC path": re.compile(r"(?<![\\])\\\\(?![.?]\\)[A-Za-z0-9_][A-Za-z0-9_.-]*\\[A-Za-z0-9_.-]+"),
    "POSIX private/local path": re.compile(r"(?<![A-Za-z0-9:/])/(?:home|Users|root|workspace|workspaces|mnt|tmp|content|kaggle|app|data|var|opt|run)(?:/|\b)", re.I),
    "file URI": re.compile(r"\bfile:/+", re.I),
    "private project desktop path": re.compile(r"(Desktop[\\/]+Research|Research[\\/]+Proj01)", re.I),
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "API/token assignment": re.compile(r"\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|secret)[\"']?\s*[:=]\s*[^\s,}]+", re.I),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private key material": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "credentialed URL": re.compile(r"\b(?:https?|s?ftp)://[^\s/:]+:[^\s/@]+@", re.I),
}
PRIVATE_PARTS = {".git", ".venv", ".ssh", ".aws", ".env", "__pycache__",
                 "chat_exports", "cover_letters", "handoff", "internal_reviewer_notes"}
PRIVATE_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".pyc", ".pyo"}
ASSET_SUFFIXES = {".zip", ".tar", ".gz", ".jpg", ".jpeg", ".png", ".webp",
                 ".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".h5"}

def main():
    failures = []
    checked = 0
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT).as_posix()
        if path.relative_to(ROOT).parts[0] == ".git":
            continue  # Root Git administration is outside the artifact.
        if path.is_symlink():
            failures.append((rel, "symlink in curated release", None))
        if any(part.lower() in PRIVATE_PARTS for part in path.relative_to(ROOT).parts) or path.suffix.lower() in PRIVATE_SUFFIXES:
            failures.append((rel, "private or transient filename", None))
        if not path.is_file():
            continue
        if path.suffix.lower() in ASSET_SUFFIXES:
            failures.append((rel, "unreviewed image/model/archive asset", None))
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        checked += 1
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append((rel, "non-UTF8 release text", None))
            continue
        for label, pattern in FORBIDDEN.items():
            for m in pattern.finditer(text):
                # Report location without echoing a potentially sensitive value.
                failures.append((rel, label, text.count("\n", 0, m.start()) + 1))
    if failures:
        for rel, label, line in failures:
            suffix = f" at line {line}" if line is not None else ""
            print(f"FAIL {rel}: {label}{suffix}")
        raise SystemExit(1)
    print(f"PASS: public-artifact hygiene scan checked {checked} text files.")

if __name__ == "__main__":
    main()
