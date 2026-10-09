"""One-shot old-sentinel DINO identity gate, then frozen-ID reserve encoding.

The two DINO I/O modules are byte copies of the previously audited encoders.
Their read_train functions are never called: SelectedTrainImages alone reads
the allowlisted TRAIN image rows. No label, request generator or model metric
is used by this program.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

from dino_io import LocalDinoB
from dino_s_io import LocalDinoS, preprocess

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
TASK = "FASTFILL_FINAL_UNSEEN_CONFIRMATION_01"
HIST = PROJECT / "current_evidence_precision_repair_02"
MODEL_CODE = HIST / "runtime_restored/model_code"
SOURCE_MANIFEST = HIST / "OFFICIAL_CODE_IDENTITY.json"
SENTINEL = HIST / "precision_preflight/OLD_SENTINELS_64.npz"
ARCHIVE = PROJECT / "capability_feasibility_01/assets/cifar-100-python.tar.gz"
WEIGHTS = {
    "S": PROJECT / "capability_feasibility_01/assets/torch_hub/checkpoints/dinov2_vits14_pretrain.pth",
    "B": PROJECT / "capability_feasibility_01/assets/torch_hub/checkpoints/dinov2_vitb14_pretrain.pth",
}
PINNED = {
    "source_manifest": "61571c423bee5117b0cf7f0580dbed5bcaf2d305d0a6050c7c489f284becd6fb",
    "sentinel": "ef5763683625965551f7bcca2200d4ca1ad09a72ea1e4cbbd9853c01d4068279",
    "archive": "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7",
    "S": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
    "B": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    "dino_io.py": "634edbae1707917e08195e4e2af9782ae85872f461438e54b3475bc5c02f4949",
    "dino_s_io.py": "d603d0fd67345555ea0b9b1a4063b97a4029c176f9e6c7c672f4026a9b52ff92",
}
PREPROCESS_SHA = "d607fc38976ff43d26193f2c7c199785ae1ba8c1ddd711d11165e5bf7d05f152"
THRESHOLDS = {"max_abs": 5e-5, "max_row_l2": 2e-4}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, obj):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def asset(path, array=None):
    p = Path(path).resolve()
    result = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}
    if array is not None:
        result["semantic"] = semantic(array)
    return result


def semantic(array):
    a = np.asarray(array)
    result = {"dtype": str(a.dtype), "shape": list(a.shape),
              "c_order_raw_sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}
    result["semantic_digest_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return result


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify(record):
    p = Path(record["path"])
    require(p.is_file() and p.stat().st_size == record["bytes"] and sha(p) == record["sha256"],
            "Frozen asset changed: " + str(p))
    return p


def asset_records(node):
    if isinstance(node, dict):
        if {"path", "bytes", "sha256"}.issubset(node):
            yield node
        else:
            for value in node.values():
                yield from asset_records(value)
    elif isinstance(node, list):
        for value in node:
            yield from asset_records(value)


def guards():
    gate_path = ROOT / "PROTOCOL_GATE.json"
    gate = read(gate_path)
    require(gate.get("status") == "PASS", "Frozen protocol identity gate must PASS")
    lock_path = ROOT / "EXECUTION_SOURCE_LOCK.json"
    lock = read(lock_path)
    require(lock.get("status") in ("FROZEN", "LOCKED", "PASS", "FROZEN_BEFORE_EXECUTION", "LOCKED_BEFORE_EXECUTION"),
            "Execution sources must be frozen before any model forward")
    records = list(asset_records(lock))
    require(records, "Execution source lock contains no asset identities")
    for record in records:
        verify(record)
    locked_paths = {os.path.normcase(os.path.realpath(row["path"])) for row in records}
    for name in ("encode_reserve.py", "dino_io.py", "dino_s_io.py", "selected_train.py", "fastfill_proxy.py", "official_transformations.py"):
        require(os.path.normcase(os.path.realpath(ROOT / "code" / name)) in locked_paths,
                "Required execution source is absent from source lock: " + name)
    public_path = ROOT / "PUBLIC_PROTOCOL.json"
    require(os.path.normcase(os.path.realpath(public_path)) in locked_paths,
            "Public frozen-ID protocol is absent from execution source lock")
    public = read(public_path)
    split = public["split"]
    selected = np.asarray(split["selected_ids"], dtype=np.int64)
    require(selected.shape == (6000,) and len(np.unique(selected)) == 6000,
            "Frozen reserve selected-ID schema differs")
    history = split["groups"]["history"]
    history_rows = np.asarray(history["rows"], dtype=np.int64)
    history_ids = np.asarray(history["global_ids"], dtype=np.int64)
    require(history_rows.shape == (4000,) and np.array_equal(history_rows, np.arange(4000, dtype=np.int64)),
            "Frozen history rows must be the original first4000")
    require(np.array_equal(history_ids, selected[history_rows]), "Frozen history IDs/order differ")
    require(int(split["stream_id"]) == 907001 and int(split["seed"]) == 907001,
            "Unique frozen stream identity differs")
    require(semantic(selected)["c_order_raw_sha256"] == split["selected_ids_sha256"],
            "Frozen selected ID raw digest differs")
    return public, selected, {"protocol_gate": asset(gate_path), "execution_source_lock": asset(lock_path),
                              "public_protocol": asset(public_path)}


def runtime():
    import torch
    import PIL
    expected = PROJECT / "capability_feasibility_01/.venv/Scripts/python.exe"
    require(os.path.normcase(os.path.realpath(sys.executable)) == os.path.normcase(os.path.realpath(expected)),
            "Only the existing historically successful CUDA interpreter is allowed")
    require(torch.__version__ == "2.11.0+cu128" and np.__version__ == "2.5.2" and torch.version.cuda == "12.8",
            "Frozen original Torch/NumPy/CUDA runtime differs; STOP without changing environment")
    require(torch.cuda.is_available() and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090",
            "Frozen original CUDA GPU unavailable")
    torch.set_num_threads(8)
    torch.manual_seed(641001)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    require(torch.get_default_dtype() == torch.float32, "DINO default dtype must be FP32")
    return {"executable": sys.executable, "python": sys.version, "torch": torch.__version__,
            "numpy": np.__version__, "pillow": PIL.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "device": "cuda", "threads": 8,
            "encoder_seed": 641001, "stream_seed_or_RNG_used_for_selection": False,
            "FP32": True, "eval": True, "TF32": False, "matmul_precision": "highest",
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "environment_installed_or_modified": False}


def verified_assets():
    for name in ("dino_io.py", "dino_s_io.py"):
        require(sha(ROOT / "code" / name) == PINNED[name], "Byte-identical DINO I/O source differs: " + name)
    paths = {"source_manifest": SOURCE_MANIFEST, "sentinel": SENTINEL, "archive": ARCHIVE, **WEIGHTS}
    result = {}
    for key, path in paths.items():
        require(sha(path) == PINNED[key], "Pinned original asset SHA256 differs: " + key)
        result[key] = asset(path)
    files = read(SOURCE_MANIFEST)
    checked = []
    for relative, identity in files.items():
        p = MODEL_CODE / relative
        require(p.is_file() and p.stat().st_size == identity["bytes"] and sha(p) == identity["sha256"],
                "Pinned DINO official source differs: " + relative)
        checked.append({"relative_path": relative, **asset(p)})
    result["model_code"] = str(MODEL_CODE.resolve())
    result["source_files"] = checked
    result["source_file_count"] = len(checked)
    result["preprocess_input_expected_raw_sha256"] = PREPROCESS_SHA
    result["archive_member_deserialized_for_hash"] = False
    return result


def save_array(path, array):
    with Path(path).open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    return asset(path, array)


def gate():
    output = ROOT / "DINO_IDENTITY_GATE.json"
    marker = ROOT / "logs/DINO_IDENTITY_ATTEMPT_STARTED.json"
    require(not output.exists() and not marker.exists(), "DINO identity gate already attempted; no retry")
    public, selected, provenance = guards()
    sources = verified_assets()
    rt = runtime()
    write_new(marker, {"task": TASK, "started_utc": utc(), "single_attempt": True,
                       "code": asset(__file__), "provenance": provenance, "sources": sources,
                       "reserve_image_access_allowed": False})
    cases = []
    arrays = {}
    access = []
    try:
        with np.load(SENTINEL, allow_pickle=False) as old:
            ids = old["ids"].astype(np.int64, copy=False).copy()
            references = {"S": old["z1"].copy(), "B": old["z2"].copy()}
        require(ids.shape == (64,) and len(np.unique(ids)) == 64 and not set(map(int, ids)) & set(map(int, selected)),
                "Historical sentinel schema or nonreserve boundary differs")
        for kind, dimension in (("S", 384), ("B", 768)):
            require(references[kind].shape == (64, dimension) and references[kind].dtype == np.float32,
                    "Historical sentinel reference schema differs: " + kind)
        from selected_train import SelectedTrainImages
        reader = SelectedTrainImages(ARCHIVE)
        images = reader.take(ids)
        require(images.shape == (64, 32, 32, 3) and images.dtype == np.uint8,
                "Allowlisted sentinel image schema differs")
        prepared = preprocess(images).numpy()
        require(semantic(prepared)["c_order_raw_sha256"] == PREPROCESS_SHA,
                "Exact historical preprocess numerical identity failed")
        arrays.update({"ids": ids, "preprocess_input": prepared, "ref_z1": references["S"],
                       "ref_z2": references["B"]})
        for kind, cls in (("S", LocalDinoS), ("B", LocalDinoB)):
            for batch in (64, 32):
                name = kind + str(batch)
                access.append({"case": name, "ids": ids.tolist(), "forward_rows": 64,
                               "batch": batch, "warmup_ids": ids[:batch].tolist(), "warmup_repeats": 3,
                               "reserve_rows": 0, "label_reads": 0})
                encoder = cls(MODEL_CODE, WEIGHTS[kind], "cuda", batch)
                try:
                    encoded, timing = encoder.encode(images)
                finally:
                    encoder.close()
                arrays[name] = encoded
                error = encoded - references[kind]
                maximum = float(np.max(np.abs(error)))
                row_l2 = float(np.max(np.linalg.norm(error, axis=1)))
                passed = maximum <= THRESHOLDS["max_abs"] and row_l2 <= THRESHOLDS["max_row_l2"]
                case = {"case": name, "model": kind, "batch": batch, "rows": 64,
                        "max_abs": maximum, "max_row_l2": row_l2, "thresholds": THRESHOLDS,
                        "pass": passed, "timing": timing, "feature_semantic": semantic(encoded)}
                cases.append(case)
                print(json.dumps({"event": "DINO_SENTINEL_CASE", **case}), flush=True)
                if not passed:
                    raise RuntimeError("Original historical sentinel threshold failed: " + name)
        require(len(cases) == 4 and all(row["pass"] for row in cases), "All four original identity cases must PASS")
        with (ROOT / "audit/DINO_GATE_ARRAYS.npz").open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        write_new(ROOT / "logs/DINO_SENTINEL_ACCESS.json", {"task": TASK, "cases": access,
                  "reserve_image_rows_read": 0, "official_TEST_read": False, "labels_read": False})
        record = {"task": TASK, "status": "PASS", "pass": True, "completed_utc": utc(),
                  "cases": cases, "thresholds": THRESHOLDS, "runtime": rt, "sources": sources,
                  "provenance": provenance, "sentinel_ids": ids.tolist(), "sentinel_ids_semantic": semantic(ids),
                  "preprocess_input": semantic(prepared), "arrays_asset": asset(ROOT / "audit/DINO_GATE_ARRAYS.npz"),
                  "sentinel_source": asset(SENTINEL),
                  "reserve_encoded": False, "reserve_pixels_read": False, "official_TEST_read": False,
                  "split_fit_pilot_request_RNG": False, "model_metrics_read": False, "retry": False}
        write_new(output, record)
        print(json.dumps({"status": record["status"], "cases": [x["case"] for x in cases]}), flush=True)
    except Exception as error:
        if arrays and not (ROOT / "audit/DINO_GATE_ARRAYS.npz").exists():
            with (ROOT / "audit/DINO_GATE_ARRAYS.npz").open("xb") as handle:
                np.savez_compressed(handle, **arrays)
        if not (ROOT / "logs/DINO_SENTINEL_ACCESS.json").exists():
            write_new(ROOT / "logs/DINO_SENTINEL_ACCESS.json", {"task": TASK, "cases": access,
                      "reserve_image_rows_read": 0, "official_TEST_read": False, "labels_read": False})
        if not output.exists():
            write_new(output, {"task": TASK, "status": "FINAL_UNSEEN_BLOCKED_DINO_IDENTITY", "pass": False,
                      "completed_utc": utc(), "cases": cases, "thresholds": THRESHOLDS,
                      "error": str(error), "traceback": traceback.format_exc(), "runtime": rt,
                      "sources": sources, "provenance": provenance, "reserve_encoded": False,
                      "reserve_pixels_read": False, "retry": False})
        raise


def encode():
    marker = ROOT / "logs/RESERVE_ENCODING_ATTEMPT_STARTED.json"
    output = ROOT / "FEATURE_CACHE_LOCK.json"
    require(not marker.exists() and not output.exists(), "Reserve encoding already attempted; no automatic retry")
    public, selected, provenance = guards()
    identity_path = ROOT / "DINO_IDENTITY_GATE.json"
    identity = read(identity_path)
    require(identity.get("status") == "PASS" and identity.get("pass") is True,
            "Four-case old-sentinel numerical PASS is required before opening reserve images")
    require(identity["thresholds"] == THRESHOLDS and [row["case"] for row in identity["cases"]] == ["S64", "S32", "B64", "B32"]
            and all(row["pass"] for row in identity["cases"]), "Historical identity gate is incomplete")
    for key in provenance:
        require(identity["provenance"][key]["sha256"] == provenance[key]["sha256"],
                "Execution/protocol binding changed after numerical identity gate")
    verify(identity["arrays_asset"])
    sources = verified_assets()
    rt = runtime()
    require(all(rt[key] == identity["runtime"][key] for key in ("executable", "torch", "numpy", "cuda", "gpu", "TF32", "matmul_precision")),
            "Reserve encoding runtime differs from successful identity gate")
    write_new(marker, {"task": TASK, "started_utc": utc(), "single_attempt": True,
                       "code": asset(__file__), "provenance": provenance,
                       "numerical_identity_gate": asset(identity_path), "global_ids": selected.tolist(),
                       "global_ids_semantic": semantic(selected), "batch": 64,
                       "models": ["S", "B"], "label_reads": 0})
    from selected_train import SelectedTrainImages
    reader = SelectedTrainImages(ARCHIVE)
    image_started = time.perf_counter()
    images = reader.take(selected)
    image_wall = time.perf_counter() - image_started
    require(images.shape == (6000, 32, 32, 3) and images.dtype == np.uint8, "Frozen6000 image schema differs")
    assets = {"global_ids": save_array(ROOT / "features/global_ids.npy", selected)}
    costs = {}
    for kind, dimension, cls in (("S", 384, LocalDinoS), ("B", 768, LocalDinoB)):
        started = time.perf_counter()
        encoder = cls(MODEL_CODE, WEIGHTS[kind], "cuda", 64)
        try:
            encoded, timing = encoder.encode(images)
        finally:
            encoder.close()
        wall = time.perf_counter() - started
        require(encoded.shape == (6000, dimension) and encoded.dtype == np.float32 and np.isfinite(encoded).all(),
                "Frozen6000 deployment feature schema differs: " + kind)
        max_norm_error = float(np.max(np.abs(np.linalg.norm(encoded, axis=1) - 1)))
        require(max_norm_error <= 2e-6, "Frozen DINO output must be L2-normalized: " + kind)
        assets[kind] = save_array(ROOT / "features" / (kind + ".npy"), encoded)
        costs[kind] = {"model": "dinov2_vits14" if kind == "S" else "dinov2_vitb14", "rows": 6000,
                       "wall_seconds": wall, "batch": 64, "warmup_rows": 192, "timing": timing,
                       "max_abs_norm_minus_one": max_norm_error, "includes_model_load": True}
        print(json.dumps({"event": "RESERVE_ENCODING_COMPLETE", "kind": kind, "cost": costs[kind]}), flush=True)
        del encoded
    write_new(ROOT / "logs/RESERVE_ENCODING_ACCESS.json", {"task": TASK, "global_ids": selected.tolist(),
              "models": ["S", "B"], "rows_per_model": 6000, "batch": 64,
              "warmup_ids_per_model": selected[:64].tolist(), "warmup_repeats": 3,
              "labels_or_eval_truth_read": False, "official_TEST_read": False,
              "split_fit_pilot_request_RNG": False, "candidate_logical_B_access": False})
    lock = {"task": TASK, "status": "FROZEN", "completed_utc": utc(), "assets": assets,
            "costs": costs, "selected_image_read_wall_seconds": image_wall,
            "global_ids": semantic(selected), "rows": 6000, "feature_dtype": "float32",
            "feature_identity": "frozen eval FP32 L2-normalized CLS", "runtime": rt,
            "sources": sources, "provenance": provenance, "numerical_identity_gate": asset(identity_path),
            "labels_or_eval_truth_read": False, "official_TEST_read": False,
            "candidate_B_access_bound_by_separate_state_receipts": True,
            "split_fit_pilot_request_RNG": False, "reencoding_or_second_stream": False}
    write_new(output, lock)
    print(json.dumps({"status": "FEATURE_CACHE_FROZEN", "assets": assets}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("gate", "encode"))
    action = parser.parse_args().action
    try:
        {"gate": gate, "encode": encode}[action]()
    except Exception as error:
        failure = ROOT / "logs" / ("DINO_" + action.upper() + "_FAILED_STOP.json")
        if not failure.exists():
            write_new(failure, {"task": TASK, "status": "FAILED_STOP", "phase": action,
                               "utc": utc(), "error": str(error), "traceback": traceback.format_exc(),
                               "automatic_retry_or_threshold_or_environment_change": False})
        if action == "gate" and not (ROOT / "DINO_IDENTITY_GATE.json").exists():
            write_new(ROOT / "DINO_IDENTITY_GATE.json", {
                "task": TASK, "status": "FINAL_UNSEEN_BLOCKED_DINO_IDENTITY", "pass": False,
                "completed_utc": utc(), "cases": [], "thresholds": THRESHOLDS,
                "error": str(error), "failure_phase": "pre_forward_identity_checks",
                "reserve_encoded": False, "reserve_pixels_read": False, "retry": False})
        raise


if __name__ == "__main__":
    main()
