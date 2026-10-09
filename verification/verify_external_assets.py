#!/usr/bin/env python3
"""Verify supplied external assets by frozen bytes/hash; never execute them."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(mapping, require_all=False):
    expected = {r["id"]: r for r in json.loads(
        (ROOT / "third_party/EXTERNAL_ASSETS.json").read_text(encoding="utf-8"))["assets"]}
    supplied = json.loads(mapping.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict) or not supplied or set(supplied) - set(expected):
        raise SystemExit("Asset map must contain known asset IDs and local filenames.")
    if require_all and set(supplied) != set(expected):
        raise SystemExit("Asset map does not cover the complete external identity inventory.")
    for identifier, filename in supplied.items():
        record = expected[identifier]
        path = Path(filename)
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise SystemExit("FAIL bytes: " + identifier)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise SystemExit("FAIL SHA256: " + identifier)
    print(f"PASS: {len(supplied)} supplied external asset identities; no scientific execution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()
    main(args.mapping, args.require_all)
