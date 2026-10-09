#!/usr/bin/env python3
"""Record frozen runtime evidence without copying private filesystem metadata."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "cifar": (1563622, "6dac21472d873856899938f57c8972d3f44fa433ec6f1f3db6a8d8a6b779045d"),
    "food": (15950893, "8250a4f98bc1a147787b40cc6ac020d07157fa2c75511d0915ce63fc58805808"),
}


def archive(path, kind):
    size, digest = EXPECTED[kind]
    if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise SystemExit("Archive identity does not match the accepted confirmation.")
    source = zipfile.ZipFile(path)
    if source.testzip() is not None:
        raise SystemExit("ZIP CRC failure.")
    return source


def member(source, suffix):
    names = [suffix] if suffix in source.namelist() else [
        n for n in source.namelist() if n.endswith("/" + suffix)]
    if len(names) != 1:
        raise SystemExit("Expected one uniquely identified frozen source member: " + suffix)
    data = source.read(names[0])
    identity = {"member": names[0], "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return json.loads(data), identity


def versions():
    result = {}
    for name in ["numpy", "scipy", "torch", "torchvision", "threadpoolctl", "Pillow"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def external_assets(source, kind):
    assets = {}
    selected = [n for n in source.namelist() if n in {
        "FEATURE_CACHE_LOCK.json", "FASTFILL_PROXY_LOCK.json", "DINO_IDENTITY_GATE.json",
        "EVALUATION_LOCK.json", "SCORE_LOCK.json", "WEIGHT_IDENTITY.json",
        "ENCODER_FEATURE_LOCK.json", "DEPLOYMENT_DATA_LOCK.json",
        "provenance/protocol_freeze/DATASET_IDENTITY.json",
        "provenance/protocol_freeze/MODEL_CANDIDATE_LOCK.json",
        "provenance/encoder_qualification/SELECTED_ENCODER_LOCK.json",
    } or n.startswith("FEATURE_CACHE_LOCKS/") and n.endswith(".json")]
    for name in selected:
        data = source.read(name)
        identity = {"member": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        def visit(value, pointer=""):
            if isinstance(value, dict):
                path = value.get("path")
                if (isinstance(path, str) and isinstance(value.get("bytes"), int)
                        and re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))
                        and path.lower().endswith((".npy", ".npz", ".pth", ".pt", ".tar.gz"))):
                    normalized = path.replace(chr(92), "/")
                    marker = ("fastfill_final_unseen_confirmation_01/" if kind == "cifar"
                              else "food101_cross_domain_revision_confirmation_01/")
                    logical = normalized.split(marker, 1)[-1] if marker in normalized else normalized.rsplit("/", 1)[-1]
                    key = (logical, value["sha256"])
                    if key not in assets:
                        assets[key] = {"id": kind + ":" + logical, "study": kind,
                                       "logical_path": logical, "bytes": value["bytes"], "sha256": value["sha256"],
                                       "source_archive_sha256": EXPECTED[kind][1],
                                       "source_member": identity, "source_pointer": pointer,
                                       "bundled": False}
                for key, item in value.items():
                    visit(item, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    visit(item, pointer + "/" + str(index))
        visit(json.loads(data))
    result = sorted(assets.values(), key=lambda r: (r["id"], r["sha256"]))
    counts = {}
    for r in result:
        counts[r["id"]] = counts.get(r["id"], 0) + 1
    for r in result:
        if counts[r["id"]] > 1:
            r["id"] += ":" + r["sha256"][:12]
    return result


def main(cifar, food):
    with archive(cifar, "cifar") as source:
        external = external_assets(source, "cifar")
        cache, cache_identity = member(source, "FEATURE_CACHE_LOCK.json")
        metadata, metadata_identity = member(source, "provenance/OLD_SOURCE_METADATA_BEFORE.json")
        encoding = {k: v for k, v in cache["runtime"].items() if k != "executable"}
        # Metadata recorded installed distribution identities before execution.
        # Preserve their exact relative names and metadata hashes; do not infer
        # that observing a directory proves an import or a clean rerun.
        distribution_records = []
        def visit(value):
            if isinstance(value, dict):
                path = value.get("path", "")
                if isinstance(path, str) and re.search(
                        r"(?:numpy|scipy|torch|torchvision|threadpoolctl)-[^/]+\.dist-info/METADATA$", path):
                    distribution_records.append({k: value[k] for k in ["path", "bytes", "sha256"] if k in value})
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(metadata)
    with archive(food, "food") as source:
        external += external_assets(source, "food")
        runtime, runtime_identity = member(source, "RUNTIME_GATE.json")
    value = {
        "schema_version": 1,
        "scope": "Historical accepted runtime evidence and observed release-tool environment; not a tested full-rerun lock.",
        "accepted_cifar_encoding": {"archive_sha256": EXPECTED["cifar"][1],
                                    "source": cache_identity, "runtime": encoding},
        "accepted_environment_distribution_inventory": {
            "archive_sha256": EXPECTED["cifar"][1], "source": metadata_identity,
            "records": distribution_records,
            "limitation": "Installed distribution metadata identities, not confirmation-phase imported-version evidence."},
        "accepted_food_encoding": {"archive_sha256": EXPECTED["food"][1], "source": runtime_identity,
                                   "actual_runtime": runtime["actual_runtime"], "settings": runtime["settings"]},
        "release_tool_environment": {"python": sys.version.split()[0], "platform": sys.platform,
                                     "observed_distribution_versions": versions(),
                                     "scientific_runners_executed": False},
        "clean_check": "See CLEAN_CHECK.json for standard-library-only verification in an isolated environment.",
        "full_end_to_end_rerun": "NOT_RUN: external assets, licensed source and relocatable identity gates required.",
    }
    (ROOT / "environment/RELEASE_ENVIRONMENT.json").write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    asset_inventory = {"schema_version": 1,
                       "scope": "External inputs and retained outputs referenced by accepted frozen identities; no payload redistribution.",
                       "assets": external}
    (ROOT / "third_party/EXTERNAL_ASSETS.json").write_text(json.dumps(asset_inventory, indent=2) + "\n", encoding="utf-8")
    print("RELEASE_ENVIRONMENT_RECORDED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-archive", required=True, type=Path)
    parser.add_argument("--food-archive", required=True, type=Path)
    args = parser.parse_args()
    main(args.cifar_archive, args.food_archive)
