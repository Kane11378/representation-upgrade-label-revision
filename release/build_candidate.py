#!/usr/bin/env python3
"""Build and verify a sealed release from tracked files or an exact sealed tree.

Git checkouts use the regular-file index as the membership whitelist. Extracted
releases must have a valid complete inventory and SHA256 manifest. Source files
are never modified; refreshed manifests are written only in the destination.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(os.path.abspath(__file__)).resolve().parents[1]
INVENTORY = "release/PUBLIC_INVENTORY.json"
MANIFEST = "release/RELEASE_FILES.sha256"
SEALS = {INVENTORY, MANIFEST}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utilities():
    sys.dont_write_bytecode = True
    manifest = runpy.run_path(str(ROOT / "verification/verify_release_manifest.py"))
    hygiene = runpy.run_path(str(ROOT / "verification/verify_public_hygiene.py"))
    return manifest, hygiene


def scan(root, hygiene):
    main = hygiene["main"]
    main.__globals__["ROOT"] = root
    main()


def members(manifest, hygiene):
    if (ROOT / ".git").exists():
        try:
            top = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                stderr=subprocess.PIPE).decode("utf-8").strip()
            if Path(top).resolve() != ROOT.resolve():
                raise ValueError("The source must be the Git checkout root.")
            raw = subprocess.check_output(
                ["git", "-C", str(ROOT), "ls-files", "--stage", "-z"],
                stderr=subprocess.PIPE)
        except (OSError, subprocess.CalledProcessError):
            raise SystemExit("Cannot read the source Git index.") from None
        paths = []
        for entry in raw.split(b"\0"):
            if not entry:
                continue
            metadata, name = entry.split(b"\t", 1)
            mode, _, stage = metadata.decode("ascii").split()
            if mode not in {"100644", "100755"} or stage != "0":
                raise SystemExit("The release index must contain only regular, resolved files.")
            paths.append(name.decode("utf-8"))
        if not paths or len(paths) != len(set(paths)):
            raise SystemExit("The source Git index is empty or has duplicate paths.")
    else:
        paths = list(manifest["verify"](ROOT))
        scan(ROOT, hygiene)
    if not SEALS <= set(paths):
        raise SystemExit("Both release seal paths must be part of the source whitelist.")
    blocked_parts = hygiene["PRIVATE_PARTS"]
    blocked_suffixes = hygiene["PRIVATE_SUFFIXES"] | hygiene["ASSET_SUFFIXES"] | {
        ".npy", ".npz", ".exe", ".dll", ".so", ".dylib",
    }
    for rel in paths:
        manifest["safe_path"](rel)
        logical = Path(rel)
        if any(part.lower() in blocked_parts for part in logical.parts) or logical.suffix.lower() in blocked_suffixes:
            raise SystemExit("The release index contains an excluded file type or directory.")
        path = ROOT / rel
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise SystemExit("A whitelisted source file is missing or unsafe.")
        if any(parent.is_symlink() for parent in path.parents if parent != ROOT and parent.is_relative_to(ROOT)):
            raise SystemExit("A whitelisted source directory is a symlink.")
    return sorted(paths)


def seal(stage, payloads):
    rows = [{"path": rel, "bytes": len(data), "sha256": sha(data)}
            for rel, data in sorted(payloads.items())]
    inventory = {"schema_version": 1, "artifact_root": "reproducibility",
                 "files": rows, "exclusions": [INVENTORY, MANIFEST]}
    payloads = dict(payloads)
    payloads[INVENTORY] = (json.dumps(inventory, indent=2) + "\n").encode("utf-8")
    payloads[MANIFEST] = ("\n".join(f"{sha(data)}  {rel}" for rel, data in sorted(payloads.items())) + "\n").encode("utf-8")
    for rel, data in sorted(payloads.items()):
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return payloads


def build(destination, refresh=False):
    if destination.is_symlink():
        raise SystemExit("Staging must not be a symlink.")
    destination = Path(os.path.abspath(destination)).resolve()
    source = ROOT.resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise SystemExit("Staging must be separate from the source tree.")
    if destination.exists() and not destination.is_dir():
        raise SystemExit("Staging must be a directory.")
    if (destination / ".git").exists() or (destination / ".git").is_symlink():
        raise SystemExit("Staging must not contain Git administration.")
    manifest, hygiene = utilities()
    paths = members(manifest, hygiene)
    payloads = {rel: (ROOT / rel).read_bytes() for rel in paths if rel not in SEALS}
    existing = destination.exists() and any(destination.iterdir())
    if existing:
        if not refresh:
            raise SystemExit("Staging must be empty unless --refresh is requested.")
        prior = manifest["verify"](destination)
        scan(destination, hygiene)
        if set(prior) - set(paths):
            raise SystemExit("Refresh cannot remove prior members; use an empty destination.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".release-build-", dir=destination.parent) as temporary:
        stage = Path(temporary).resolve()
        if stage.parent != destination.parent.resolve():
            raise SystemExit("Temporary staging is outside the selected parent.")
        sealed = seal(stage, payloads)
        manifest["verify"](stage)
        scan(stage, hygiene)
        checked = subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(stage / "verification/run_all.py")],
            cwd=stage, capture_output=True, text=True, encoding="utf-8")
        if checked.returncode != 0 or "ALL_LIGHTWEIGHT_CHECKS_PASS" not in checked.stdout:
            raise SystemExit("The staged artifact failed lightweight verification.")
        if any((ROOT / rel).read_bytes() != data for rel, data in payloads.items()):
            raise SystemExit("A source file changed while the release was being built.")
        destination.mkdir(parents=True, exist_ok=True)
        for rel in sorted(sealed):
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(stage / rel, target)
        manifest["verify"](destination)
        print(checked.stdout, end="")
    print(f"CANDIDATE_SEALED: {len(sealed)} files; source files unchanged.")
    return {rel: sha(data) for rel, data in sealed.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--refresh", action="store_true", help="Refresh a verified previous release without removing members")
    args = parser.parse_args()
    build(args.destination, args.refresh)
