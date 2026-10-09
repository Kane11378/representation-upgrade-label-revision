#!/usr/bin/env python3
"""Verify complete release inventory and SHA256 identities, including staging."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = "release/PUBLIC_INVENTORY.json"
MANIFEST = "release/RELEASE_FILES.sha256"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(value):
    require(isinstance(value, str) and value, "empty or non-string release path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and "\\" not in value and ":" not in value,
            f"non-relative release path: {value}")
    require(path.as_posix() == value and all(p not in {"", ".", ".."} for p in value.split("/")),
            f"non-canonical release path: {value}")
    return value


def file_map(root):
    files = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.relative_to(root).parts[0] == ".git":
            continue  # Root Git administration is outside the artifact.
        require(not path.is_symlink(), f"symlink is not a release file: {rel}")
        require("__pycache__" not in path.parts and path.suffix.lower() not in {".pyc", ".pyo"},
                f"transient Python cache in release: {rel}")
        if path.is_file():
            safe_path(rel)
            payload = path.read_bytes()
            files[rel] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    return files


def verify(root):
    files = file_map(root)
    require(INVENTORY in files and MANIFEST in files, "release inventory or SHA256 manifest missing")
    inventory = json.loads((root / INVENTORY).read_text(encoding="utf-8"))
    require(inventory.get("schema_version") == 1, "unsupported inventory schema")
    require(inventory.get("artifact_root") == "reproducibility", "unexpected artifact root identity")
    require(set(inventory.get("exclusions", [])) == {INVENTORY, MANIFEST}, "inventory exclusions changed")
    rows = inventory.get("files")
    require(isinstance(rows, list), "inventory files must be a list")
    recorded = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}, "invalid inventory row")
        rel = safe_path(row["path"])
        require(rel not in recorded, f"duplicate inventory path: {rel}")
        require(type(row["bytes"]) is int and row["bytes"] >= 0 and
                isinstance(row["sha256"], str) and HEX64.fullmatch(row["sha256"]),
                f"invalid file identity: {rel}")
        recorded[rel] = {"bytes": row["bytes"], "sha256": row["sha256"]}
    expected = {path: identity for path, identity in files.items() if path not in {INVENTORY, MANIFEST}}
    require(recorded.keys() == expected.keys(),
            f"inventory membership mismatch; missing={sorted(expected.keys() - recorded.keys())}; "
            f"unexpected={sorted(recorded.keys() - expected.keys())}")
    for rel, identity in expected.items():
        require(recorded[rel] == identity, f"inventory byte/hash mismatch: {rel}")

    hashes = {}
    for line in (root / MANIFEST).read_text(encoding="utf-8").splitlines():
        require(len(line) > 66 and line[64:66] == "  " and HEX64.fullmatch(line[:64]),
                "malformed SHA256 manifest line")
        rel = safe_path(line[66:])
        require(rel not in hashes, f"duplicate SHA256 manifest path: {rel}")
        hashes[rel] = line[:64]
    require(set(hashes) == set(files) - {MANIFEST}, "SHA256 manifest membership mismatch")
    for rel, digest in hashes.items():
        require(digest == files[rel]["sha256"], f"SHA256 manifest mismatch: {rel}")
    return files


def main():
    files = verify(ROOT)
    stage = ROOT.parent / "public_release"
    if ROOT.name == "reproducibility" and stage.exists():
        stage_files = verify(stage)
        require(files == stage_files, "public staging tree differs from the curated reproducibility tree")
    print(f"PASS: exact public inventory and SHA256 identities verified for {len(files)} release files.")


if __name__ == "__main__":
    main()
