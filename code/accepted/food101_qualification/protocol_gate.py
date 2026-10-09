"""Write-once standard-library protocol/asset gate; no image decoding or ML imports."""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
FREEZE = WORKSPACE / "food101_cross_domain_protocol_freeze_01"
ZIP = FREEZE / "deliverables/011_food101_cross_domain_protocol_freeze_01_20261006.zip"
EXPECTED_ZIP = {"bytes": 10652341, "members": 81, "sha256": "5730eaf2d5c9a69777b7c8f5b6000b5af1360abfad97a81abf45958d3bc20a3d"}
REQUIRED = ["DATASET_IDENTITY.json", "SIX_FOLD_LOCK.json", "MODEL_CANDIDATE_LOCK.json", "QUALIFICATION_RULE_LOCK.json", "NO_ACCESS_AUDIT.json"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": hash_file(path)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def copy_exact(src, dst):
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    b = src.read_bytes()
    with dst.open("xb") as f:
        f.write(b)
    record = identity(dst)
    record["source_path"] = str(src.resolve())
    require(record["bytes"] == len(b) and record["sha256"] == hashlib.sha256(b).hexdigest(), "copy identity mismatch")
    return record


def source_records(value):
    if isinstance(value, dict):
        if {"original_absolute_path", "copy_relative_path", "bytes", "sha256"}.issubset(value):
            yield value
        else:
            for child in value.values():
                yield from source_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from source_records(child)


def version_literals(path):
    values = {}
    for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                    values[target.id] = node.value.value
    return values


def discover_weights(model):
    models = {"resnet18": model["old_encoder"], **model["candidates"]}
    filenames = {enc: rec["weight_url"].rsplit("/", 1)[-1] for enc, rec in models.items()}
    found = {enc: [] for enc in filenames}
    reverse = {filename: enc for enc, filename in filenames.items()}
    exclusions = {".git", ".venv", "assets", "features", "metrics", "datasets", "data", "node_modules", "__pycache__"}
    for base, dirs, files in os.walk(WORKSPACE):
        dirs[:] = [d for d in dirs if d not in exclusions]
        for filename in files:
            if filename in reverse:
                found[reverse[filename]].append(Path(base) / filename)
    cache_roots = [Path.home() / ".cache/torch/hub/checkpoints"]
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        cache_roots.append(Path(torch_home) / "hub/checkpoints")
    for directory in cache_roots:
        if directory.is_dir():
            for enc, filename in filenames.items():
                path = directory / filename
                if path.is_file():
                    found[enc].append(path)
    records = {}
    for enc, paths in found.items():
        paths = sorted(set(p.resolve() for p in paths), key=str)
        records[enc] = []
        for path in paths:
            rec = identity(path)
            rec.update({"published_url": models[enc]["weight_url"], "published_sha256_prefix": models[enc]["url_sha256_prefix"]})
            rec["published_prefix_match"] = rec["sha256"].startswith(rec["published_sha256_prefix"])
            records[enc].append(rec)
    write_json(ROOT / "audit/WEIGHT_DISCOVERY.json", {
        "status": "ALL_FOUR_CACHED_WEIGHTS_FOUND" if all(records.values()) else "SOME_OFFICIAL_WEIGHTS_NOT_CACHED",
        "created_utc": utc(), "encoders": records,
        "searched_roots": [str(WORKSPACE), *map(str, cache_roots)],
        "workspace_directory_exclusions": sorted(exclusions),
        "only_exact_weight_filenames_read": list(filenames.values()),
        "downloaded": False, "model_loaded": False, "forward_or_probe_executed": False,
    })
    return records


def main():
    started, t0 = utc(), time.perf_counter()
    own_lock = read_json(ROOT / "audit/PROTOCOL_GATE_SOURCE_LOCK.json")
    source_id = identity(__file__)
    require(own_lock["files"]["code/protocol_gate.py"]["sha256"] == source_id["sha256"], "gate source lock mismatch")
    require(own_lock["files"]["code/protocol_gate.py"]["bytes"] == source_id["bytes"], "gate source bytes mismatch")
    contract = read_json(ROOT / "ARTIFACT_CONTRACT.json")
    require(platform.python_version() == contract["runtime"]["python"], "FOOD101_QUALIFICATION_RUNTIME_IDENTITY_MISMATCH: Python")
    zid = identity(ZIP)
    require(zid["bytes"] == EXPECTED_ZIP["bytes"] and zid["sha256"] == EXPECTED_ZIP["sha256"], "protocol ZIP fixed identity mismatch")
    manifest = read_json(FREEZE / "MANIFEST.json")
    require(manifest["status"] == "FROZEN", "protocol manifest status mismatch")
    zipped_records, disk_records = {}, {}
    with zipfile.ZipFile(ZIP) as archive:
        members = archive.namelist()
        require(len(members) == len(set(members)) == EXPECTED_ZIP["members"], "protocol ZIP member count or duplicates mismatch")
        require(archive.testzip() is None, "protocol ZIP CRC failure")
        require(set(members) == set(manifest["files"]) | {"MANIFEST.json"}, "protocol ZIP/manifest membership mismatch")
        require(archive.read("MANIFEST.json") == (FREEZE / "MANIFEST.json").read_bytes(), "protocol manifest disk/ZIP mismatch")
        for relative, expected in manifest["files"].items():
            path = FREEZE / relative
            require(path.resolve().is_relative_to(FREEZE.resolve()), "unsafe manifest path")
            payload = archive.read(relative)
            zipped = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            require(zipped == expected, f"protocol ZIP manifest mismatch: {relative}")
            disk = identity(path)
            require(disk["bytes"] == expected["bytes"] and disk["sha256"] == expected["sha256"], f"protocol disk manifest mismatch: {relative}")
            zipped_records[relative], disk_records[relative] = zipped, disk
    zid.update({"members": len(members), "crc_pass": True})
    status = read_json(FREEZE / "PROTOCOL_STATUS.json")
    require(status["status"] == "FOOD101_CROSS_DOMAIN_PROTOCOL_FROZEN_AUDIT_PASS", "protocol status mismatch")
    require(status["encoder_qualification_executed"] is False and status["revision_executed"] is False, "protocol has executed results")
    locks = {name: read_json(FREEZE / name) for name in REQUIRED}
    model, rule, data, six, noaccess = (locks[n] for n in ["MODEL_CANDIDATE_LOCK.json", "QUALIFICATION_RULE_LOCK.json", "DATASET_IDENTITY.json", "SIX_FOLD_LOCK.json", "NO_ACCESS_AUDIT.json"])
    require(data["status"] == "FOOD101_TEST_POOL_IDENTITY_PASS" and data["official_train_split_used"] is False, "dataset status mismatch")
    require(six["status"] == "FROZEN" and six["split_selection_uses_models_or_performance"] is False, "fold lock status mismatch")
    require(model["status"] == rule["status"] == "FROZEN_FOR_LATER_EXECUTION", "candidate/rule status mismatch")
    require(model["qualification_executed"] is False and rule["qualification_executed"] is False, "old qualification results present")
    require(noaccess["status"] == "NO_FORBIDDEN_ACCESS_PASS", "original no-access status mismatch")
    require(rule["train_fold"] == 0 and rule["validation_fold"] == 1 and rule["classes"] == 101 and rule["probe"]["ridge"] == 1.0, "qualification rule mismatch")
    require(rule["selection"]["exact_candidate_tie_priority"] == contract["candidate_tie_priority"], "tie priority mismatch")
    runtime_records, copies = [], []
    seen = set()
    for record in source_records(model):
        original = Path(record["original_absolute_path"])
        relative = record["copy_relative_path"]
        if relative in seen:
            continue
        seen.add(relative)
        real = identity(original)
        frozen = identity(FREEZE / relative)
        require(all(real[k] == frozen[k] == record[k] for k in ["bytes", "sha256"]), f"FOOD101_QUALIFICATION_RUNTIME_IDENTITY_MISMATCH: {original}")
        runtime_records.append({"original": real, "frozen": frozen, "frozen_record": record})
        copies.append(copy_exact(FREEZE / relative, ROOT / relative))
    require(len(runtime_records) == 8, "expected 8 distinct frozen runtime source records")
    runtime = model["runtime_source_versions"]
    torch_literal = version_literals(Path(runtime["torch_version_source"]["original_absolute_path"]))
    vision_literal = version_literals(Path(runtime["torchvision_version_source"]["original_absolute_path"]))
    require(torch_literal["__version__"] == runtime["torch"] == contract["runtime"]["torch"], "FOOD101_QUALIFICATION_RUNTIME_IDENTITY_MISMATCH: torch literal")
    require(torch_literal["git_version"] == runtime["torch_git_version"], "FOOD101_QUALIFICATION_RUNTIME_IDENTITY_MISMATCH: torch git")
    require(vision_literal["__version__"] == runtime["torchvision"] == contract["runtime"]["torchvision"], "FOOD101_QUALIFICATION_RUNTIME_IDENTITY_MISMATCH: torchvision literal")
    require(vision_literal["git_version"] == runtime["torchvision_git_version"], "FOOD101_QUALIFICATION_RUNTIME_IDENTITY_MISMATCH: torchvision git")
    for filename in [*REQUIRED, "PROTOCOL_STATUS.json", "MANIFEST.json"]:
        copies.append(copy_exact(FREEZE / filename, ROOT / "provenance/protocol_freeze" / filename))
    require(len(data["class_to_idx"]) == 101 and set(data["class_to_idx"].values()) == set(range(101)), "class mapping mismatch")
    allowed_ids, access_records, fold_counts = set(), [], {}
    groups = {f"fold{fold}": six["folds"][str(fold)] for fold in (0, 1)}
    image_base = (FREEZE / "assets/food-101/images").resolve()
    for name, group in groups.items():
        images, rows = group["images"], group["rows"]
        require(len(images) == len(rows) == 4242, f"{name} count mismatch")
        require(rows == [im["row"] for im in images], f"{name} row order mismatch")
        counts = Counter(im["class_id"] for im in images)
        require(set(counts) == set(range(101)) and set(counts.values()) == {42}, f"{name} class balance mismatch")
        for im in images:
            require(im == data["images"][im["row"]], f"{name} original dataset row mismatch")
            require(im["image_id"] not in allowed_ids, "fold0/fold1 duplicate or intersection")
            allowed_ids.add(im["image_id"])
            require(data["class_to_idx"][im["class_name"]] == im["class_id"], "clean label mapping mismatch")
            path = Path(im["path"])
            require(path.resolve().is_relative_to(image_base), "image path outside clean frozen test pool")
            real = identity(path)
            require(real["bytes"] == im["bytes"] and real["sha256"] == im["sha256"], f"{name} raw image identity mismatch: {im['image_id']}")
            access_records.append({"fold": int(name[-1]), "image_id": im["image_id"], "row": im["row"], "class_id": im["class_id"], **real, "access": "RAW_BYTES_SHA256_ONLY", "decoded": False})
        fold_counts[name] = {"images": len(images), "classes": len(counts), "per_class": 42, "raw_byte_identities_verified": len(images)}
    write_json(ROOT / "logs/PROTOCOL_GATE_RAW_IMAGE_ACCESS.json", {"status": "ONLY_FOLD0_FOLD1_RAW_IMAGE_HASH_ACCESS_PASS", "created_utc": utc(), "access_count": len(access_records), "allowed_folds": [0, 1], "image_decodes": 0, "records": access_records})
    data_lock = {
        "status": "QUALIFICATION_DATA_LOCK_PASS", "created_utc": utc(),
        "folds": groups, "class_to_idx": data["class_to_idx"], "classes": data["classes"],
        "source_identity": {name: disk_records[name] for name in REQUIRED},
        "source_locks": {name: disk_records[name] for name in REQUIRED},
        "protocol_zip": zid, "canonical_group_order_preserved": True,
        "raw_image_access_log": identity(ROOT / "logs/PROTOCOL_GATE_RAW_IMAGE_ACCESS.json"),
        "image_decode_count": 0, "permitted_performance_folds": [0, 1],
        "folds2_3_4_5_image_payloads_accessed": False,
        "official_train_split_used": False, "provenance_copies": copies,
    }
    write_json(ROOT / "QUALIFICATION_DATA_LOCK.json", data_lock)
    weights = discover_weights(model)
    gate = {
        "status": "PASS", "started_utc": started, "completed_utc": utc(),
        "protocol_zip": zid, "manifest_identity": identity(FREEZE / "MANIFEST.json"),
        "manifest_record_count": len(manifest["files"]), "all_manifest_files_disk_and_zip_identical": True,
        "required_locks": {name: disk_records[name] for name in REQUIRED},
        "original_protocol_status": status, "runtime_source_records": runtime_records,
        "runtime_versions_static": {"python": platform.python_version(), "torch": torch_literal["__version__"], "torchvision": vision_literal["__version__"]},
        "runtime_models_not_imported": True, "provenance_copies": copies,
        "fold_counts": fold_counts, "qualification_data_lock": identity(ROOT / "QUALIFICATION_DATA_LOCK.json"),
        "raw_image_access_log": identity(ROOT / "logs/PROTOCOL_GATE_RAW_IMAGE_ACCESS.json"),
        "source_lock": identity(ROOT / "audit/PROTOCOL_GATE_SOURCE_LOCK.json"),
        "execution_source": source_id, "artifact_contract_identity": identity(ROOT / "ARTIFACT_CONTRACT.json"),
        "weight_discovery_identity": identity(ROOT / "audit/WEIGHT_DISCOVERY.json"),
        "cached_weight_counts": {enc: len(values) for enc, values in weights.items()},
        "source_and_asset_access_scope": "Protocol ZIP/manifest metadata and frozen static source bytes; only fold0/fold1 raw JPEG bytes for SHA256; exact official cached weight payload bytes for SHA256.",
        "image_decodes": 0, "performance_computations": 0,
        "forbidden_methods_imported_or_executed": False,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    write_json(ROOT / "PROTOCOL_GATE.json", gate)
    print(json.dumps({"status": gate["status"], "fold_counts": fold_counts, "protocol_zip": zid, "cached_weights": weights}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
