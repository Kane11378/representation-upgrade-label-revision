"""Independent NumPy audit; execute only after root authorization and evaluation.

No candidate/core imports, no RNG, no model or image loading. Frozen choices
are read directly, never regenerated. Existing arrays are reconstructed with
independent np.linalg.solve only after all candidates are hash locked.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TASK = "FASTFILL_FINAL_UNSEEN_CONFIRMATION_01"
PROTOCOL_ZIP_SHA = "744ebb41dccaa2a99e16c878b01c2c1dd72b4a1c2850f906a294c50f7d77355c"
CHECKPOINT_SHA = "0ac692e1e9dd0994e89a3f40fbc165a34cd097503ded8ac56cea0249712538f8"
CORE_SHA = "0b28ae97c4fcd65eb901938e06a8642a899e5c32208b239da7b4ac64101d6bd7"
STAGES = ["before_R1", "after_refresh_R1", "after_R1", "before_R2", "after_refresh_R2", "after_R2"]
CANDIDATES = ["mean", "paired025", "sample_current"]
METHODS = CANDIDATES + ["full_target_ridge"]
FIELDS = ["proxy", "observed", "labels", "versions", "known", "certainty", "pilot", "pilot_rows", "Q_mean", "H_mean", "Q_sample", "H_sample"]
TOLERANCE = 1e-8
CHECKS = []
READS = []
MAXIMUM = {name: {"maximum_absolute": 0.0, "maximum_norm_relative": 0.0, "largest_absolute_context": ""} for name in ["state", "QH", "W", "score", "metric", "frontend"]}


def require(condition, name, detail=None):
    row = {"check": name, "passed": bool(condition)}
    if detail is not None:
        row["detail"] = detail
    CHECKS.append(row)
    if not condition:
        raise AssertionError(name + (": " + str(detail) if detail is not None else ""))


def read_json(path):
    path = Path(path).resolve()
    READS.append({"path": str(path), "mode": "json"})
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha_file(path):
    path = Path(path).resolve()
    READS.append({"path": str(path), "mode": "raw_sha256_only"})
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def asset(record, root, name):
    require(isinstance(record, dict) and "path" in record and "sha256" in record, name + " asset schema")
    path = Path(record["path"])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    require(path.is_file(), name + " exists")
    require(sha_file(path) == record["sha256"], name + " bytes SHA256 identity")
    if "bytes" in record:
        require(path.stat().st_size == int(record["bytes"]), name + " bytes count identity")
    return path


def semantic(array):
    array = np.ascontiguousarray(array)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "c_order_raw_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def semantic_record_same(actual, expected, name):
    # Prior protocol locks use dtype.str and a longer raw-byte key. The new
    # execution interface uses the concise key; aliases only name the same hash.
    raw_key = next((key for key in ["c_order_raw_sha256", "c_order_raw_bytes_sha256"] if key in expected), None)
    require(raw_key is not None, name + " C-order raw hash exists")
    dtype = np.dtype(expected["dtype"]).str
    require(actual["dtype"] == dtype and actual["shape"] == expected["shape"] and actual["c_order_raw_sha256"] == expected[raw_key], name + " semantic identity")
    if "semantic_digest_sha256" in expected:
        digest_fields = {key: value for key, value in expected.items() if key != "semantic_digest_sha256"}
        digest = hashlib.sha256(json.dumps(digest_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        require(digest == expected["semantic_digest_sha256"], name + " normalized semantic digest identity")


def load_npy_record(record, root, name):
    path = asset(record, root, name)
    READS.append({"path": str(path), "mode": "saved_numeric_npy"})
    value = np.load(path, allow_pickle=False)
    require(isinstance(value, np.ndarray) and value.dtype.kind in "biuf", name + " numeric ndarray")
    require(np.isfinite(value).all(), name + " finite")
    semantic_record_same(semantic(value), record["semantic"], name)
    return value


def npz_arrays(path):
    path = Path(path).resolve()
    READS.append({"path": str(path), "mode": "saved_numeric_npz"})
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def exact_array(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, name + " shape")
    require(actual.dtype == expected.dtype, name + " dtype", {"actual": str(actual.dtype), "expected": str(expected.dtype)})
    require(np.array_equal(actual, expected), name + " exact values")


def near(actual, expected, name, kind):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, name + " shape")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), name + " finite")
    difference = actual - expected
    maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
    denominator = float(np.linalg.norm(expected))
    relative = float(np.linalg.norm(difference) / denominator) if denominator else (0.0 if maximum == 0 else None)
    record = MAXIMUM[kind]
    if maximum > record["maximum_absolute"]:
        record["maximum_absolute"] = maximum
        record["largest_absolute_context"] = name
    if relative is not None:
        record["maximum_norm_relative"] = max(record["maximum_norm_relative"], relative)
    require(maximum <= TOLERANCE, name + " absolute numerical identity", {"maximum_absolute": maximum, "norm_relative": relative, "tolerance": TOLERANCE})


def rms(value):
    return float(np.sqrt(np.mean(np.square(value))))


def relative(error, reference):
    return float(np.linalg.norm(error) / max(float(np.linalg.norm(reference)), 1e-15))


def add_bias(value):
    value = np.asarray(value, dtype=np.float64)
    return np.concatenate([value, np.ones((len(value), 1), dtype=np.float64)], axis=1)


def class_sums(features, labels):
    # This class-wise reduction is independent of core's np.add.at path.
    result = np.empty((features.shape[1], 100), dtype=np.float64)
    for label in range(100):
        result[:, label] = features[labels == label].sum(axis=0)
    return result


def independent_matrices(proxy, observed, labels, known, certainty, pilot):
    n, d = proxy.shape
    means = proxy.copy()
    means[known] = observed[known]
    q_mean = means.T @ means + np.eye(d, dtype=np.float64)
    h_mean = class_sums(means, labels)
    weights = np.zeros(n, dtype=np.float64)
    if known.all():
        q_sample = q_mean.copy()
        weights[:] = 1.0
    else:
        remaining = ~certainty
        population_count = int(remaining.sum())
        remaining_pilot = pilot[remaining[pilot]]
        require(len(remaining_pilot) > 0, "sample remaining pilot nonempty")
        feature_pilot = observed[remaining_pilot, :-1]
        mean = feature_pilot.mean(axis=0)
        residual = feature_pilot - mean
        covariance = residual.T @ residual / len(remaining_pilot)
        covariance = 0.9 * covariance + 0.1 * np.trace(covariance) / (d - 1) * np.eye(d - 1)
        second = np.empty((d, d), dtype=np.float64)
        second[:-1, :-1] = covariance + mean[:, None] * mean[None, :]
        second[:-1, -1] = mean
        second[-1, :-1] = mean
        second[-1, -1] = 1.0
        q_sample = observed[certainty].T @ observed[certainty] + population_count * second + np.eye(d, dtype=np.float64)
        weights[certainty] = 1.0
        weights[remaining_pilot] = population_count / len(remaining_pilot)
    h_sample = class_sums(observed * weights[:, None], labels)
    return q_mean, h_mean, q_sample, h_sample


def check_locked_files(lock, root, context):
    require("files" in lock and isinstance(lock["files"], dict), context + " locked files present")
    for name, record in lock["files"].items():
        if isinstance(record, str):
            path = root / name
            require(path.resolve().is_relative_to(root.resolve()), context + " path stays inside task: " + name)
            require(sha_file(path) == record, context + " file SHA: " + name)
        else:
            path = asset(record, root, context + " file " + name)
            require(path.is_relative_to(root.resolve()), context + " asset stays inside task: " + name)


def check_source_rng(root, source_lock):
    for name in source_lock["files"]:
        if not name.endswith(".py"):
            continue
        record = source_lock["files"][name]
        path = Path(record["path"]) if isinstance(record, dict) else root / name
        if not path.is_absolute():
            path = root / path
        text = path.read_text(encoding="utf-8-sig")
        READS.append({"path": str(path.resolve()), "mode": "source_AST"})
        tree = ast.parse(text, filename=str(path))
        forbidden = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = ast.unparse(node.func)
            if callee.endswith((".default_rng", ".RandomState", ".permutation", ".shuffle")) or callee in {"default_rng", "RandomState"}:
                forbidden.append({"line": node.lineno, "call": callee})
        require(not forbidden, "no RNG split/evidence/request-generating call: " + name, forbidden)
    core_path = root / "code" / "current_core.py"
    require(sha_file(core_path) == CORE_SHA, "exact unchanged current_core source")
    require(sha_file(Path(__file__).resolve()) == (source_lock["files"]["code/independent_confirmation_audit.py"]["sha256"] if isinstance(source_lock["files"]["code/independent_confirmation_audit.py"], dict) else source_lock["files"]["code/independent_confirmation_audit.py"]), "this independent auditor source frozen")


def check_protocol(protocol_root):
    archive = protocol_root / "deliverables" / "011_fastfill_final_unseen_protocol_freeze_01_20261006.zip"
    require(archive.stat().st_size == 1490420, "accepted protocol ZIP bytes")
    require(sha_file(archive) == PROTOCOL_ZIP_SHA, "accepted protocol ZIP SHA256")
    with zipfile.ZipFile(archive) as zipped:
        require(len(zipped.infolist()) == 46, "accepted protocol ZIP member count")
        require(zipped.testzip() is None, "accepted protocol ZIP CRC")
    manifest = read_json(protocol_root / "MANIFEST.json")
    names = ["RESERVE_IDENTITY.json", "FINAL_STREAM_SPLIT_LOCK.json", "FIT_PILOT_LOCK.json", "REQUESTS_LOCK.json", "METHOD_LOCK.json", "SOURCE_PROTOCOL_IDENTITY.json", "CLAIM_FREEZE.md"]
    for name in names:
        record = manifest["files"][name]
        path = protocol_root / name
        require(path.stat().st_size == record["bytes"] and sha_file(path) == record["sha256"], "accepted exact protocol artifact: " + name)
    reserve = read_json(protocol_root / names[0])
    split = read_json(protocol_root / names[1])
    evidence = read_json(protocol_root / names[2])
    requests = read_json(protocol_root / names[3])
    method = read_json(protocol_root / names[4])
    source = read_json(protocol_root / names[5])
    require(reserve["status"] == "PASS" and source["status"] == "PASS", "accepted identity and recovered semantics PASS")
    require(split["stream_id"] == 907001 and split["seed"] == 907001 and requests["stream_id"] == 907001, "one exact locked stream 907001")
    selected = np.asarray(split["selected_ids"], dtype=np.int64)
    require(len(selected) == 6000 and len(np.unique(selected)) == 6000, "locked 6000 unique global IDs")
    require(np.all((selected >= 0) & (selected < 50000)), "official TRAIN namespace, never TEST")
    for group, sl in [("history", slice(0, 4000)), ("arrival_E", slice(4000, 5000)), ("evaluation", slice(5000, 6000))]:
        exact_array(np.asarray(split["groups"][group]["global_ids"], dtype=np.int64), selected[sl], "locked group IDs " + group)
    fit = np.asarray(evidence["fit"]["rows"], dtype=np.int64)
    pilot = np.asarray(evidence["pilot"]["rows"], dtype=np.int64)
    require(len(fit) == len(pilot) == 512 and len(np.unique(np.concatenate([fit, pilot]))) == 1024, "fixed disjoint 512 fit and 512 pilot")
    require(np.all((fit >= 0) & (fit < 4000)) and np.all((pilot >= 0) & (pilot < 4000)), "history-only fit/pilot")
    exact_array(selected[fit], np.asarray(evidence["fit"]["global_ids"], dtype=np.int64), "fit rows/global IDs binding")
    exact_array(selected[pilot], np.asarray(evidence["pilot"]["global_ids"], dtype=np.int64), "pilot rows/global IDs binding")
    require(method["backend"]["methods"] == ["mean", "paired025"], "only frozen mean and paired025 methods")
    require(method["backend"]["alpha"] == .25 and method["backend"]["ridge"] == 1 and method["backend"]["shrink"] == .1, "alpha/ridge/shrink fixed")
    require(method["frontend"]["checkpoint_sha256"] == CHECKPOINT_SHA and method["backend"]["current_core_sha256"] == CORE_SHA, "frozen model and core identities")
    for name in ["post_normalization", "pilot_bias"]:
        require(method["frontend"][name] is False, "prohibited frontend change absent " + name)
    require(method["evidence"]["sigma_selection"] is False and method["evidence"]["common"] is True, "common evidence with no sigma selection")
    require(method["additional_methods_or_hybrid_combinations_allowed"] is False and method["additional_streams_or_seeds_allowed"] is False, "no other method/hybrid/stream allowed")
    require(requests["requests"]["R1"]["count"] == 819 and requests["requests"]["R2"]["count"] == 1228, "exact locked 819 and 1228 request counts")
    return split, evidence, requests, selected, fit, pilot


def gate_and_assets(root):
    protocol_gate = read_json(root / "PROTOCOL_GATE.json")
    require(protocol_gate["status"] == "PASS", "execution protocol gate PASS")
    source_lock = read_json(root / "EXECUTION_SOURCE_LOCK.json")
    check_locked_files(source_lock, root, "execution source lock")
    check_source_rng(root, source_lock)
    numerical = read_json(root / "DINO_IDENTITY_GATE.json")
    require(numerical["status"] == "PASS", "DINO-S/B numerical identity gate PASS")
    require({row["case"] for row in numerical["cases"]} == {"S64", "S32", "B64", "B32"}, "exact four legacy sentinel cases")
    require(len(numerical["cases"]) == 4, "four sentinel cases only")
    for case in numerical["cases"]:
        require(case["pass"] is True and 0 <= case["max_abs"] <= 5e-5 and 0 <= case["max_row_l2"] <= 2e-4, "original sentinel limits: " + case["case"])
        require(case["batch"] == int(case["case"][1:]), "sentinel original batch: " + case["case"])
    sentinel_arrays = npz_arrays(asset(numerical["arrays_asset"], root, "sentinel saved arrays"))
    require(set(sentinel_arrays) == {"ids", "preprocess_input", "ref_z1", "ref_z2", "S64", "S32", "B64", "B32"}, "sentinel saved array keys")
    original_sentinel = npz_arrays(asset(numerical["sentinel_source"], root, "historical nonreserve sentinel"))
    exact_array(sentinel_arrays["ids"], original_sentinel["ids"].astype(np.int64), "historical sentinel IDs exact")
    exact_array(sentinel_arrays["ref_z1"], original_sentinel["z1"], "historical S sentinel reference exact")
    exact_array(sentinel_arrays["ref_z2"], original_sentinel["z2"], "historical B sentinel reference exact")
    for case in numerical["cases"]:
        value = sentinel_arrays[case["case"]]
        reference = sentinel_arrays["ref_z1" if case["case"][0] == "S" else "ref_z2"]
        require(value.dtype == reference.dtype == np.dtype("float32") and value.shape == reference.shape == (64, 384 if case["case"][0] == "S" else 768), "sentinel saved FP32 shape " + case["case"])
        error = value - reference
        maximum = float(np.max(np.abs(error)))
        maximum_l2 = float(np.max(np.linalg.norm(error, axis=1)))
        require(maximum <= 5e-5 and maximum_l2 <= 2e-4, "independently checked original sentinel gate " + case["case"])
        near(case["max_abs"], maximum, "saved sentinel max_abs " + case["case"], "metric")
        near(case["max_row_l2"], maximum_l2, "saved sentinel max_row_l2 " + case["case"], "metric")
        semantic_record_same(semantic(value), case["feature_semantic"], "sentinel output " + case["case"])
    sources = numerical["sources"]
    pinned_sources = {"source_manifest": "61571c423bee5117b0cf7f0580dbed5bcaf2d305d0a6050c7c489f284becd6fb", "sentinel": "ef5763683625965551f7bcca2200d4ca1ad09a72ea1e4cbbd9853c01d4068279", "archive": "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7", "S": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9", "B": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"}
    for name, expected_sha in pinned_sources.items():
        require(sources[name]["sha256"] == expected_sha, "pinned original DINO source identity " + name)
        asset(sources[name], root, "DINO original source " + name)
    for record in sources["source_files"]:
        asset(record, root, "DINO pinned source file " + record["relative_path"])
    for name, expected_sha in {"dino_io.py": "634edbae1707917e08195e4e2af9782ae85872f461438e54b3475bc5c02f4949", "dino_s_io.py": "d603d0fd67345555ea0b9b1a4063b97a4029c176f9e6c7c672f4026a9b52ff92"}.items():
        require(sha_file(root / "code" / name) == expected_sha, "historical exact DINO preprocess/CLS/L2 source " + name)
    require(semantic(sentinel_arrays["preprocess_input"])["c_order_raw_sha256"] == "d607fc38976ff43d26193f2c7c199785ae1ba8c1ddd711d11165e5bf7d05f152", "historical exact preprocessed sentinel numeric input")
    runtime = numerical["runtime"]
    require(runtime["FP32"] is True and runtime["eval"] is True and runtime["TF32"] is False and runtime["matmul_precision"] == "highest", "historical FP32/eval/TF32/highest runtime")
    require(runtime["environment_installed_or_modified"] is False and runtime["stream_seed_or_RNG_used_for_selection"] is False, "no environment change or protocol RNG in encoder")
    for flag in ["reserve_encoded", "reserve_pixels_read", "official_TEST_read", "split_fit_pilot_request_RNG", "model_metrics_read", "retry"]:
        require(numerical[flag] is False, "identity phase excluded " + flag)
    cache_lock = read_json(root / "FEATURE_CACHE_LOCK.json")
    s = load_npy_record(cache_lock["assets"]["S"], root, "reserve S features")
    b = load_npy_record(cache_lock["assets"]["B"], root, "reserve B features")
    ids = load_npy_record(cache_lock["assets"]["global_ids"], root, "reserve feature IDs")
    require(not np.intersect1d(sentinel_arrays["ids"], ids).size, "old sentinel IDs excluded frozen reserve")
    require(cache_lock["numerical_identity_gate"]["sha256"] == sha_file(root / "DINO_IDENTITY_GATE.json"), "reserve encoded only after exact numerical gate")
    require(s.shape == (6000, 384) and b.shape == (6000, 768), "S/B exact 6k dimensions")
    require(s.dtype == b.dtype == np.dtype("float32"), "S/B FP32 cache dtype")
    require(ids.shape == (6000,) and ids.dtype == np.dtype("int64"), "cache global ID int64 vector")
    for model, features in [("S", s), ("B", b)]:
        require(float(np.max(np.abs(np.linalg.norm(features, axis=1) - 1.0))) <= 2e-6, "normalized CLS features: " + model)
    proxy_lock = read_json(root / "FASTFILL_PROXY_LOCK.json")
    proxy = load_npy_record(proxy_lock["assets"]["proxy"], root, "history FastFill proxy")
    sigma = load_npy_record(proxy_lock["assets"]["sigma"], root, "history descriptive sigma")
    require(proxy.shape == (4000, 768) and proxy.dtype == np.dtype("float32"), "proxy FP32 4000x768")
    require(sigma.size == 4000 and sigma.dtype == np.dtype("float32"), "sigma FP32 4000 values")
    checkpoint_record = proxy_lock["checkpoint"]
    checkpoint_path = asset(checkpoint_record, root, "frozen FastFill epoch79 raw byte identity")
    require(checkpoint_path.stat().st_size == 60189229 and checkpoint_record["sha256"] == CHECKPOINT_SHA, "exact epoch79 checkpoint bytes/SHA")
    for flag in ["post_normalization", "pilot_bias", "sigma_evidence_selection"]:
        require(proxy_lock[flag] is False, "proxy forbidden operation absent: " + flag)
    require(proxy_lock["eval"] is True and proxy_lock["frozen"] is True and proxy_lock["BN_unchanged"] is True, "FastFill eval mode and parameters/BN frozen")
    require(proxy_lock["batch_size"] == 1024, "FastFill fixed batch1024")
    build = read_json(root / "BUILD_LOCK.json")
    require(build["status"] == "ALL_CANDIDATES_LOCKED_BEFORE_EVALUATION", "build locked before any performance")
    require(build["stream_id"] == 907001 and build["methods"] == CANDIDATES and build["stages"] == STAGES, "one stream all 18 candidate heads frozen")
    require(build["performance_computed"] is False, "no performance computed during build")
    check_locked_files(build, root, "candidate build lock")
    evaluation = read_json(root / "EVALUATION_LOCK.json")
    require(evaluation["build_lock_sha256"] == sha_file(root / "BUILD_LOCK.json"), "evaluation bound to exact build lock")
    require(evaluation["source_lock_sha256"] == sha_file(root / "EXECUTION_SOURCE_LOCK.json"), "evaluation bound to exact source lock")
    check_locked_files(evaluation, root, "completed evaluation lock")
    require((root / "BUILD_LOCK.json").stat().st_mtime_ns <= (root / "EVALUATION_LOCK.json").stat().st_mtime_ns, "build physically preceded completed evaluation")
    return cache_lock, proxy_lock, ids, s, b, proxy, sigma, source_lock, numerical, build, evaluation


def simulate_protocol(requests, selected, fit, pilot):
    labels = np.asarray(requests["stage_labels"]["before_R1"]["labels"], dtype=np.int64)
    versions = np.zeros(4000, dtype=np.int64)
    known = np.zeros(4000, dtype=bool)
    known[fit] = True
    known[pilot] = True
    certainty = np.zeros(4000, dtype=bool)
    certainty[fit] = True
    snapshots = {}
    access_events = {}

    def snapshot(stage):
        locked = requests["stage_labels"][stage]
        require(len(labels) == locked["active_rows"], "active stage size " + stage)
        exact_array(labels, np.asarray(locked["labels"], dtype=np.int64), "frozen active labels " + stage)
        exact_array(versions, np.asarray(locked["versions"], dtype=np.int64), "frozen active label versions " + stage)
        snapshots[stage] = {"labels": labels.copy(), "versions": versions.copy(), "known": known.copy(), "certainty": certainty.copy()}

    snapshot("before_R1")
    for event in ["R1", "R2"]:
        if event == "R2":
            e_labels = np.asarray(requests["stage_labels"]["before_R2"]["labels"], dtype=np.int64)[4000:5000]
            labels = np.concatenate([labels, e_labels])
            versions = np.concatenate([versions, np.zeros(1000, dtype=np.int64)])
            known = np.concatenate([known, np.ones(1000, dtype=bool)])
            certainty = np.concatenate([certainty, np.ones(1000, dtype=bool)])
            snapshot("before_R2")
        rows = np.asarray(requests["requests"][event]["rows"], dtype=np.int64)
        require(len(rows) == requests["requests"][event]["count"] and len(np.unique(rows)) == len(rows), "locked distinct request rows " + event)
        fresh = rows[~known[rows]]
        already = rows[known[rows]]
        access_events[event] = {"requested_rows": rows.tolist(), "fresh_rows": fresh.tolist(), "already_known_rows": already.tolist(), "new_history_rows": fresh[fresh < 4000].tolist(), "global_ids": selected[rows].tolist()}
        known[rows] = True
        certainty[rows] = True
        snapshot("after_refresh_" + event)
        entries = requests["requests"][event]["entries"]
        require(len(entries) == len(rows), "every locked request has release entry " + event)
        for index, entry in enumerate(entries):
            row = int(entry["row"])
            require(row == int(rows[index]) and entry["request_index"] == index and entry["global_id"] == int(selected[row]), "request entry row/order/ID " + event + "/" + str(index))
            require(entry["old_label"] == int(labels[row]) and entry["expected_version"] == int(versions[row]) == 0, "release old label/version " + event + "/" + str(index))
            require(entry["feature_version"] == "v2" and entry["feature_refresh_stage"] == "after_refresh_" + event and entry["label_release_stage"] == "after_refresh_" + event and entry["label_apply_stage"] == "after_" + event, "strict refresh-before-release " + event + "/" + str(index))
            labels[row] = int(entry["new_label"])
            versions[row] = int(entry["resulting_version"])
        snapshot("after_" + event)
    require(np.all(snapshots["after_R2"]["versions"] <= 1), "no duplicated release/version progression")
    return snapshots, access_events


def check_receipt(receipt, stage, snapshot, proxy, observed, pilot):
    require(receipt["known_rows"] == np.flatnonzero(snapshot["known"]).tolist(), stage + " receipt exact known rows")
    require(receipt["certainty_rows"] == np.flatnonzero(snapshot["certainty"]).tolist(), stage + " receipt exact certainty rows")
    require(receipt["pilot_rows"] == pilot.tolist(), stage + " receipt original pilot order")
    require(receipt["count"] == int(snapshot["known"].sum()), stage + " receipt known count")
    for field, value in [("labels", snapshot["labels"]), ("versions", snapshot["versions"]), ("proxy", proxy), ("observed_exact", observed)]:
        semantic_record_same(semantic(value), receipt[field], stage + " receipt " + field)
    require(np.count_nonzero(observed[~snapshot["known"]]) == 0, stage + " observed receipt contains zero unknown payload")
    require(receipt["sigma_selection"] is False, stage + " no sigma selection receipt")


def read_csv(path):
    path = Path(path).resolve()
    READS.append({"path": str(path), "mode": "saved_metric_csv"})
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def cosine_class_retrieval(queries, gallery, y_queries, y_gallery):
    queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-15)
    gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-15)
    ordering = np.argsort(-(queries @ gallery.T), axis=1, kind="stable")
    relevant = y_gallery[ordering] == y_queries[:, None]
    denominator = relevant.sum(axis=1)
    precision = np.cumsum(relevant, axis=1) / (1 + np.arange(len(gallery)))
    average_precision = (precision * relevant).sum(axis=1) / np.maximum(denominator, 1)
    return {"retrieval_top1": float(relevant[:, 0].mean()), "retrieval_map": float(average_precision[denominator > 0].mean()), "queries_with_relevant": int((denominator > 0).sum())}


def verify_frontend(root, proxy, b, fit, pilot, requests, eval_truth, expected_stage_metrics):
    hold_mask = np.ones(4000, dtype=bool)
    hold_mask[fit] = False
    hold_mask[pilot] = False
    hold = np.flatnonzero(hold_mask)
    require(len(hold) == 2976, "fresh fixed hold exactly history minus fit/pilot")
    gallery_labels = np.asarray(requests["stage_labels"]["before_R1"]["labels"], dtype=np.int64).copy()
    for event in ["R1", "R2"]:
        for entry in requests["requests"][event]["entries"]:
            if entry["row"] < 4000:
                gallery_labels[entry["row"]] = entry["new_label"]
    z = proxy[hold].astype(np.float64)
    target = b[hold].astype(np.float64)
    z_centered = z - z.mean(axis=0)
    t_centered = target - target.mean(axis=0)
    z_norm = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-15)
    t_norm = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-15)
    unique, counts = np.unique(z, axis=0, return_counts=True)
    result = {"n": 2976, "feature_rms": rms(z - target), "cosine_similarity": float(np.sum(z_norm * t_norm, axis=1).mean()), "centered_output_rms": rms(z_centered), "centered_target_rms": rms(t_centered), "centered_output_target_variation_ratio": rms(z_centered) / rms(t_centered), "covariance_trace_ratio": float(np.sum(z_centered * z_centered) / np.sum(t_centered * t_centered)), "exact_distinct_rows": len(unique), "duplicate_rows": len(z) - len(unique), "duplicate_groups": int(np.sum(counts > 1)), "largest_duplicate_group": int(counts.max()), "before_R1_mean_accuracy": expected_stage_metrics[("before_R1", "mean")]["accuracy"]}
    result.update(cosine_class_retrieval(b[5000:].astype(np.float64), z, eval_truth, gallery_labels[hold]))
    qualification = read_json(root / "FRONTEND_QUALIFICATION.json")
    actual = qualification["metrics"]
    require(qualification["hold_rows"] == hold.tolist(), "frontend qualification exact fixed hold rows")
    for field, value in result.items():
        require(field in actual, "frontend secondary metric retained " + field)
        if field in {"n", "exact_distinct_rows", "duplicate_rows", "duplicate_groups", "largest_duplicate_group", "queries_with_relevant"} or field == "before_R1_mean_accuracy":
            require(actual[field] == value, "frontend exact secondary " + field)
        else:
            near(actual[field], value, "frontend secondary " + field, "frontend")
    return result


def audit(root, protocol_root):
    split, evidence, requests, selected, fit, pilot = check_protocol(protocol_root)
    cache_lock, proxy_lock, ids, s, b, proxy_raw, sigma, source_lock, numerical, build, evaluation = gate_and_assets(root)
    exact_array(ids, selected, "cache IDs equal exact ordered frozen reserve")
    snapshots, access_events = simulate_protocol(requests, selected, fit, pilot)
    states = npz_arrays(root / "built" / "STATES_ARRAYS.npz")
    heads = npz_arrays(root / "built" / "HEADS.npz")
    require(set(states) == {stage + "__" + field for stage in STAGES for field in FIELDS}, "exact saved state key set")
    require(set(heads) == {stage + "__" + method for stage in STAGES for method in CANDIDATES}, "18 candidate heads only")
    access = read_json(root / "ACCESS_RECEIPTS.json")
    require(set(access["stages"]) == set(STAGES), "exact six access stages")
    initial = np.concatenate([fit, pilot])
    require(len(np.unique(initial)) == 1024, "initial unique fit plus pilot without order changes")
    require(access["initial"]["rows"] == initial.tolist() and access["initial"]["global_ids"] == selected[initial].tolist() and access["initial"]["count"] == 1024, "initial exact B unique fit union pilot")
    require(access["append_E"]["rows"] == list(range(4000, 5000)) and access["append_E"]["global_ids"] == selected[4000:5000].tolist() and access["append_E"]["count"] == 1000, "append E authoritative exact target receipt")
    for event in ["R1", "R2"]:
        for key, expected in access_events[event].items():
            require(access["events"][event][key] == expected, "frozen target access event " + event + "/" + key)
    require(all(row < 4000 for row in access_events["R2"]["fresh_rows"]), "R2 E rows never duplicate target access")
    target_augmented = add_bias(b)
    proxy_augmented = add_bias(proxy_raw)
    reconstructed_w = {}
    for stage in STAGES:
        snap = snapshots[stage]
        n = len(snap["labels"])
        proxy = proxy_augmented if n == 4000 else np.vstack([proxy_augmented, target_augmented[4000:5000]])
        observed = np.zeros_like(proxy)
        observed[snap["known"]] = target_augmented[:n][snap["known"]]
        pilot_mask = np.zeros(n, dtype=bool)
        pilot_mask[pilot] = True
        expectations = {"proxy": proxy, "observed": observed, "labels": snap["labels"], "versions": snap["versions"], "known": snap["known"], "certainty": snap["certainty"], "pilot": pilot_mask, "pilot_rows": pilot}
        for field, expected in expectations.items():
            actual = states[stage + "__" + field]
            exact_array(actual, expected, "stage exact state " + stage + "/" + field)
        require(np.count_nonzero(states[stage + "__observed"][~snap["known"]]) == 0, "unknown target payload strictly absent " + stage)
        receipts = access["stages"][stage]
        require(set(receipts) == set(CANDIDATES), "only fixed candidate receipts " + stage)
        require(receipts["mean"] == receipts["paired025"] == receipts["sample_current"], "common receipts identical " + stage)
        check_receipt(receipts["mean"], stage, snap, proxy, observed, pilot)
        q_mean, h_mean, q_sample, h_sample = independent_matrices(proxy, observed, snap["labels"], snap["known"], snap["certainty"], pilot)
        for field, expected in [("Q_mean", q_mean), ("H_mean", h_mean), ("Q_sample", q_sample), ("H_sample", h_sample)]:
            near(states[stage + "__" + field], expected, "stage independently reconstructed " + stage + "/" + field, "QH")
        w_mean = np.linalg.solve(q_mean, h_mean)
        w_sample = np.linalg.solve(q_sample, h_sample)
        for method, expected in [("mean", w_mean), ("sample_current", w_sample), ("paired025", .75 * w_mean + .25 * w_sample)]:
            near(heads[stage + "__" + method], expected, "independent np.linalg.solve W " + stage + "/" + method, "W")
            reconstructed_w[(stage, method)] = expected
    final_known = np.flatnonzero(snapshots["after_R2"]["known"])
    require(access["final_candidate_unique_target_rows"] == len(final_known), "final logical unique B target rows")
    require(len(final_known) < 5000, "candidate never has all active target rows")
    eval_truth = np.asarray(requests["evaluation_truth"]["labels"], dtype=np.int64)
    require(len(eval_truth) == 1000 and requests["evaluation_truth"]["access"] == "evaluator-only; never released as training labels", "clean eval truth evaluator-only")
    exact_array(np.asarray(requests["evaluation_truth"]["global_ids"], dtype=np.int64), selected[5000:], "truth locked evaluation global IDs")
    full_heads = npz_arrays(root / "evaluation" / "HEADS_FULL.npz")
    require(set(full_heads) == {stage + "__full_target_ridge" for stage in STAGES}, "full reference evaluator six heads only")
    scores = npz_arrays(root / "evaluation" / "SCORES.npz")
    predictions = npz_arrays(root / "evaluation" / "PREDICTIONS.npz")
    required_keys = {stage + "__" + method for stage in STAGES for method in METHODS}
    require(set(scores) == set(predictions) == required_keys, "all six stages and four score/prediction methods retained")
    v = target_augmented[5000:]
    reconstructed_scores = {}
    stage_metrics = {}
    for stage in STAGES:
        active_labels = snapshots[stage]["labels"]
        feature_matrix = target_augmented[:len(active_labels)]
        q_full = feature_matrix.T @ feature_matrix + np.eye(769, dtype=np.float64)
        h_full = class_sums(feature_matrix, active_labels)
        w_full = np.linalg.solve(q_full, h_full)
        near(full_heads[stage + "__full_target_ridge"], w_full, "full reference active stage labels W " + stage, "W")
        reconstructed_w[(stage, "full_target_ridge")] = w_full
        reference = v @ w_full
        for method in METHODS:
            key = stage + "__" + method
            score = v @ reconstructed_w[(stage, method)]
            reconstructed_scores[(stage, method)] = score
            near(scores[key], score, "independent score " + key, "score")
            expected_prediction = score.argmax(axis=1)
            exact_array(predictions[key], expected_prediction, "independent predictions " + key)
            stage_metrics[(stage, method)] = {"accuracy": float(np.mean(expected_prediction == eval_truth)), "score_rms": rms(score - reference), "weight_rel": relative(reconstructed_w[(stage, method)] - w_full, w_full), "known": int(snapshots[stage]["known"].sum())}
    saved_stages = read_csv(root / "STAGE_METRICS.csv")
    require(len(saved_stages) == 24 and {(row["stage"], row["method"]) for row in saved_stages} == set(stage_metrics), "all stage secondary cells retained")
    for row in saved_stages:
        key = (row["stage"], row["method"])
        expected = stage_metrics[key]
        for field in ["accuracy", "known"]:
            actual = int(row[field]) if field == "known" else float(row[field])
            require(actual == expected[field], "exact secondary " + "/".join(key) + "/" + field)
        for field in ["score_rms", "weight_rel"]:
            near(float(row[field]), expected[field], "secondary " + "/".join(key) + "/" + field, "metric")
    event_metrics = {}
    for event in ["R1", "R2"]:
        full_response = reconstructed_scores[("after_" + event, "full_target_ridge")] - reconstructed_scores[("after_refresh_" + event, "full_target_ridge")]
        for method in METHODS:
            response = reconstructed_scores[("after_" + event, method)] - reconstructed_scores[("after_refresh_" + event, method)]
            total = reconstructed_scores[("after_" + event, method)] - reconstructed_scores[("before_" + event, method)]
            event_metrics[(event, method)] = {"request_count": requests["requests"][event]["count"], "response_error_rms": rms(response - full_response), "reference_response_rms": rms(full_response), "response_rel": relative(response - full_response, full_response), "total_maintenance_error_rms": rms(total - full_response), "reference_total_action_rms": rms(full_response), "total_action_rel": relative(total - full_response, full_response)}
    saved_events = read_csv(root / "REQUESTS_ALL.csv")
    require(len(saved_events) == 8 and {(row["event"], row["method"]) for row in saved_events} == set(event_metrics), "all event controls and secondary metrics retained")
    for row in saved_events:
        key = (row["event"], row["method"])
        for field, expected in event_metrics[key].items():
            if field == "request_count":
                require(int(row[field]) == expected, "exact frozen event count " + "/".join(key))
            else:
                near(float(row[field]), expected, "independent event metric " + "/".join(key) + "/" + field, "metric")
    primary = read_json(root / "PRIMARY_RESULT.json")
    passes = []
    for event in ["R1", "R2"]:
        mean = event_metrics[(event, "mean")]["response_rel"]
        paired = event_metrics[(event, "paired025")]["response_rel"]
        direction = paired < mean
        passes.append(direction)
        signed = (mean - paired) / mean if mean != 0 else None
        result = primary["events"][event]
        near(result["mean_response_rel"], mean, "primary mean response " + event, "metric")
        near(result["paired025_response_rel"], paired, "primary paired response " + event, "metric")
        require(result["direction_pass"] is direction, "primary strict direction " + event)
        if signed is None:
            require(result["signed_relative_reduction"] is None, "zero mean signed reduction undefined " + event)
        else:
            near(result["signed_relative_reduction"], signed, "primary signed relative reduction " + event, "metric")
    status = "FINAL_UNSEEN_PRIMARY_CONFIRMED_BOTH" if all(passes) else "FINAL_UNSEEN_PRIMARY_PARTIAL" if any(passes) else "FINAL_UNSEEN_PRIMARY_NOT_CONFIRMED"
    require(primary["status"] == status, "primary status solely two prespecified directions")
    frontend = verify_frontend(root, proxy_raw, b, fit, pilot, requests, eval_truth, stage_metrics)
    require(not any(name == "current_core" or name.startswith(("torch", "PIL", "torchvision")) for name in sys.modules), "independent audit loaded no candidate/model/image modules")
    return {"primary_status": status, "stream_id": 907001, "stream_count": 1, "stage_count": 6, "candidate_head_count": 18, "candidate_unique_target_rows": len(final_known), "target_access": {"initial": 1024, "R1_fresh": len(access_events["R1"]["fresh_rows"]), "append_E": 1000, "R2_fresh": len(access_events["R2"]["fresh_rows"])}, "maximum_discrepancies": MAXIMUM, "independent_frontend_secondary": frontend, "independent_request_metrics": [{"event": event, "method": method, **metrics} for (event, method), metrics in event_metrics.items()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--root-authorized-after-evaluation", action="store_true")
    args = parser.parse_args()
    if not args.root_authorized_after_evaluation:
        parser.error("Root authorization after build/evaluation is mandatory; this script must not run before performance is complete.")
    root = args.root.resolve()
    out = root / "audit" / "INDEPENDENT_CONFIRMATION_AUDIT.json"
    if out.exists():
        raise FileExistsError("Once-only independent audit output already exists")
    require((root / "BUILD_LOCK.json").is_file() and (root / "EVALUATION_LOCK.json").is_file(), "build and completed evaluation required before any audit array read")
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = audit(root, args.protocol_root.resolve())
        report = {"task": TASK, "status": "PASS", "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(), "root_authorized_after_evaluation": True, "independent_of_candidate_solve": True, "rng_calls": 0, "official_TEST_used": False, "model_inference_or_training": False, "comparison_absolute_tolerance": TOLERANCE, "checks_passed": len(CHECKS), "checks": CHECKS, "reads": READS, **result}
        write_json_new(out, report)
        print(json.dumps({"status": "PASS", "checks_passed": len(CHECKS), "primary_status": result["primary_status"], "maximum_discrepancies": MAXIMUM}, ensure_ascii=False), flush=True)
    except Exception as error:
        failure = {"task": TASK, "status": "FINAL_UNSEEN_INVALID_INDEPENDENT_AUDIT", "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(), "error_type": type(error).__name__, "reason": str(error), "checks": CHECKS, "reads": READS, "maximum_discrepancies": MAXIMUM, "STOP": True, "no_scientific_repair_or_rerun_authorized": True}
        write_json_new(out, failure)
        raise


if __name__ == "__main__":
    main()
