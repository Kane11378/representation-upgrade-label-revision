"""Evaluate all four frozen clean fold0->fold1 caches, then apply the exact rule.

No image loading or torchvision model execution is permitted in this stage.
Every all-eight cache and all relevant source/data/weight identity is checked
before any fit, score, confusion table, accuracy, or selection computation.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
ENCODERS = ("resnet18", "convnext_tiny", "swin_t", "vit_b_16")
CANDIDATES = ("convnext_tiny", "swin_t", "vit_b_16")
DIMS = {"resnet18": 512, "convnext_tiny": 768, "swin_t": 768, "vit_b_16": 768}
COUNT = 4242
CLASSES = 101


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    p = Path(path).resolve()
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)}


def check_identity(record):
    actual = identity(record["path"])
    require(actual["bytes"] == record["bytes"] and actual["sha256"] == record["sha256"],
            "Locked file identity differs: " + record["path"])
    return actual


def write_json_once(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")


class EvaluationGuard:
    def __init__(self, weight_paths):
        self.attempts = []
        self.weight_whitelist = {str(Path(p).resolve()).casefold() for p in weight_paths}

    def audit(self, event, args):
        reason = None
        p = None
        if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"):
            reason = "Network and subprocess execution forbidden during locked ridge evaluation"
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            lower_parts = [part.casefold() for part in p.parts]
            if p.suffix.casefold() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"):
                reason = "Ridge stage cannot open any image"
            if any("current_core" in part or "procrustes" in part for part in lower_parts) or any(
                    part in ("r1", "r2", "r1_results", "r2_results") for part in lower_parts):
                reason = "Forbidden historical/core/revision artifact"
            if p.is_relative_to(PROJECT) and not p.is_relative_to(ROOT):
                allowed_source = p.is_relative_to(PROJECT / ".venv")
                allowed_freeze_metadata = p.is_relative_to(PROJECT / "food101_cross_domain_protocol_freeze_01")
                allowed_weight = str(p).casefold() in self.weight_whitelist and p.suffix.casefold() == ".pth"
                if not allowed_source and not allowed_freeze_metadata and not allowed_weight:
                    reason = "Prior unrelated workspace artifact outside authorized inputs"
                if p.suffix.casefold() in (".py", ".pyc", ".pyo") and not allowed_source:
                    reason = "Prior workspace executable code forbidden"
        if reason:
            self.attempts.append({"event": event, "path": str(p) if p else None, "reason": reason, "utc": utc()})
            raise PermissionError(reason)


def validate_sources_and_gates(build):
    execution = read_json(ROOT / "EXECUTION_SOURCE_LOCK_V2.json")
    require(execution["status"] == "FROZEN_BEFORE_FEATURE_OR_PERFORMANCE_EXECUTION", "Execution source status")
    require(set(execution["original_source_locks"]) == {"EXECUTION_SOURCE_LOCK.json", "AUDIT_SOURCE_LOCK.json"}, "Both original source locks preserved")
    for filename, expected in execution["original_source_locks"].items():
        original_path = ROOT / filename
        require(original_path.stat().st_size == expected["bytes"] and sha(original_path) == expected["sha256"], "Original source lock modified: " + filename)
        for relative, record in read_json(original_path)["files"].items():
            original_file = (ROOT / relative).resolve()
            require(original_file.is_relative_to(ROOT) and original_file.stat().st_size == record["bytes"] and
                    sha(original_file) == record["sha256"], "Original source modified: " + relative)
    require("code/encode_qualification_v2.py" in execution["files"] and "code/ridge_qualification_v2.py" in execution["files"], "Source-lock coverage")
    for rel, rec in execution["files"].items():
        path = (ROOT / rel).resolve()
        require(path.is_relative_to(ROOT) and path.stat().st_size == rec["bytes"] and sha(path) == rec["sha256"], "Execution source identity: " + rel)
    require(check_identity(build["execution_source_lock_identity"]) == identity(ROOT / "EXECUTION_SOURCE_LOCK_V2.json"), "Build execution source-lock identity")
    runtime = read_json(ROOT / "RUNTIME_GATE.json")
    require(runtime["status"] == "PASS" and runtime["runtime_status"] == "FOOD101_QUALIFICATION_RUNTIME_IDENTITY_PASS", "Runtime gate must pass")
    actual = runtime["actual_runtime"]
    require(actual["python"] == platform.python_version() == "3.12.14" and actual["torch"] == "2.14.1+cpu" and actual["torchvision"] == "0.29.1+cpu", "Frozen runtime gate versions")
    for rec in runtime["sources"]:
        require(rec["identity_match"] and rec["bytes"] == rec["expected_bytes"] and rec["sha256"] == rec["expected_sha256"], "Source gate expected identity")
        check_identity(rec)
    require(check_identity(build["runtime_gate_identity"]) == identity(ROOT / "RUNTIME_GATE.json"), "Build runtime gate identity")
    require(runtime["execution_source_lock_identity"] == build["execution_source_lock_identity"], "Runtime/build source consistency")
    model = read_json(ROOT / "provenance" / "protocol_freeze" / "MODEL_CANDIDATE_LOCK.json")
    check_identity(runtime["model_candidate_lock_identity"])
    weights = read_json(ROOT / "WEIGHT_IDENTITY.json")
    require(weights["status"] == "ALL_FOUR_WEIGHT_IDENTITIES_PASS" and set(weights["encoders"]) == set(ENCODERS), "All-four weights gate")
    require(check_identity(build["weight_identity"]) == identity(ROOT / "WEIGHT_IDENTITY.json"), "Build weight gate identity")
    records = {"resnet18": model["old_encoder"], **model["candidates"]}
    for name in ENCODERS:
        r = weights["encoders"][name]
        frozen = records[name]
        check_identity(r)
        require(r["published_url"] == frozen["weight_url"] and r["published_sha256_prefix"] == frozen["url_sha256_prefix"] and
                r["published_prefix_match"] is True and r["sha256"].startswith(frozen["url_sha256_prefix"]), "Frozen weight identity: " + name)
    return model, weights


def semantic_arrays(arrays):
    import numpy as np
    rec = {key: {"shape": list(array.shape), "dtype": array.dtype.str,
                 "c_order_raw_sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()}
           for key, array in sorted(arrays.items())}
    packed = json.dumps(rec, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return rec, hashlib.sha256(packed).hexdigest()


def all_eight_preconditions():
    import numpy as np
    build_path = ROOT / "FEATURE_BUILD_LOCK.json"
    build = read_json(build_path)
    require(build["status"] == "ALL_EIGHT_FEATURE_CACHES_FROZEN_BEFORE_PERFORMANCE" and
            build["no_accuracy_computed_during_build"] is True and build["forbidden_attempts"] == [], "Build-all lock must precede any performance")
    require(set(build["caches"]) == {n + "_fold" + str(f) for n in ENCODERS for f in (0, 1)}, "Exactly eight caches locked")
    model, weights = validate_sources_and_gates(build)
    data = read_json(ROOT / "QUALIFICATION_DATA_LOCK.json")
    require(data["status"] == "QUALIFICATION_DATA_LOCK_PASS" and set(data["folds"]) == {"fold0", "fold1"}, "Clean qualification data gate")
    require(check_identity(build["data_lock_identity"]) == identity(ROOT / "QUALIFICATION_DATA_LOCK.json"), "Build data lock identity")
    require(check_identity(build["protocol_gate_identity"]) == identity(ROOT / "PROTOCOL_GATE.json"), "Build audited protocol gate identity")
    protocol = read_json(ROOT / "PROTOCOL_GATE.json")
    require(protocol["status"] == "PASS" and protocol["image_decodes"] == 0 and protocol["performance_computations"] == 0, "Audited protocol source gate")
    mapping = data["class_to_idx"]
    require(len(mapping) == CLASSES and set(mapping.values()) == set(range(CLASSES)), "Exact 101-class map")
    values = {}
    for name in ENCODERS:
        for fold in (0, 1):
            key = name + "_fold" + str(fold)
            record = build["caches"][key]
            cache_lock_path = ROOT / "FEATURE_CACHE_LOCKS" / (key + ".json")
            require(read_json(cache_lock_path) == record, "Per-cache lock must equal all-eight lock record")
            require(record["status"] == "FEATURE_CACHE_IDENTITY_PASS" and record["encoder"] == name and record["fold"] == fold, "Per-cache status/name/fold")
            expected_path = (ROOT / "features" / name / ("fold" + str(fold) + ".npz")).resolve()
            require(Path(record["path"]).resolve() == expected_path, "Locked feature path")
            check_identity(record)
            with np.load(expected_path, allow_pickle=False) as loaded:
                require(set(loaded.files) == {"features", "image_ids", "labels", "rows"}, "Exact cache keys")
                arrays = {k: loaded[k].copy() for k in loaded.files}
            require(arrays["features"].shape == (COUNT, DIMS[name]) and arrays["features"].dtype.str == "<f4", "Feature dimensions and FP32")
            require(arrays["image_ids"].shape == (COUNT,) and arrays["image_ids"].dtype.str == "<U64", "Cache image IDs schema")
            require(all(arrays[k].shape == (COUNT,) and arrays[k].dtype.str == "<i8" for k in ("labels", "rows")), "Cache labels/rows schema")
            group = data["folds"]["fold" + str(fold)]
            images = group["images"]
            require(len(images) == COUNT and group["rows"] == [r["row"] for r in images], "Canonical fold rows")
            require(np.array_equal(arrays["image_ids"], np.array([r["image_id"] for r in images], dtype="<U64")) and
                    np.array_equal(arrays["labels"], np.array([r["class_id"] for r in images], dtype="<i8")) and
                    np.array_equal(arrays["rows"], np.array([r["row"] for r in images], dtype="<i8")), "Exact canonical IDs, labels, rows, order")
            require(np.array_equal(np.bincount(arrays["labels"], minlength=CLASSES), np.full(CLASSES, 42)), "42/class clean fold")
            require(np.isfinite(arrays["features"]).all(), "Finite normalized cache")
            norms = np.linalg.norm(arrays["features"].astype(np.float64), axis=1)
            require(np.all(np.abs(norms - 1.0) <= 0.000002) and float(norms.min()) == record["row_norm_min"] and
                    float(norms.max()) == record["row_norm_max"], "Locked FP32 row-L2 cache")
            array_record, semantic = semantic_arrays(arrays)
            require(array_record == record["arrays"] and semantic == record["semantic_digest"], "Cache semantic identity")
            require(dt.datetime.fromisoformat(record["created_utc"]) <= dt.datetime.fromisoformat(build["created_utc"]), "Cache lock creation must precede build lock")
            require(expected_path.stat().st_mtime_ns <= build_path.stat().st_mtime_ns and
                    cache_lock_path.stat().st_mtime_ns <= build_path.stat().st_mtime_ns, "Cache and lock mtimes must precede build lock")
            log_rec = record["access_log"]
            check_identity(log_rec)
            log = read_json(log_rec["path"])
            expected_access = [{"image_id": r["image_id"], "row": r["row"], "class_id": r["class_id"],
                                "path": str(Path(r["path"]).resolve()), "bytes": r["bytes"], "sha256": r["sha256"]} for r in images]
            require(log["status"] == "QUALIFICATION_IMAGE_ACCESS_PASS" and log["images"] == expected_access and
                    log["count"] == COUNT and log["unique_count"] == COUNT and log["forbidden_attempts"] == [], "Clean raw image access log exactness")
            require(log["encoder"] == name and log["fold"] == fold and log_rec["image_count"] == COUNT, "Access log encoder and fold")
            frozen_record = model["old_encoder"] if name == "resnet18" else model["candidates"][name]
            require(record["hook_definition"] == frozen_record["feature_hook"], "Frozen exact preclassifier hook")
            require(record["transform"]["enum"] == frozen_record["weights_enum"] and record["transform"]["url"] == frozen_record["weight_url"], "Frozen official weight transform identity")
            values[key] = arrays
    for fold in (0, 1):
        require(len(set(values["resnet18_fold" + str(fold)]["image_ids"].tolist())) == COUNT, "Unique clean fold IDs")
    require(not set(values["resnet18_fold0"]["image_ids"].tolist()) & set(values["resnet18_fold1"]["image_ids"].tolist()), "Disjoint clean folds")
    # This marker is written after all eight verification passes and before any fit.
    evaluation_started = utc()
    require(dt.datetime.fromisoformat(evaluation_started) > dt.datetime.fromisoformat(build["created_utc"]), "Evaluation must start strictly after build lock")
    write_json_once(ROOT / "logs" / "EVALUATION_PRECONDITIONS.json",
                    {"status": "ALL_EIGHT_REVERIFIED_BEFORE_ANY_RIDGE_PERFORMANCE", "created_utc": evaluation_started,
                     "feature_build_lock_identity": identity(build_path), "all_cache_keys": list(values),
                     "checks": ["file_hash", "semantic_hash", "exact_ids", "clean_labels", "canonical_rows", "canonical_order",
                                "shape_dtype", "finite", "row_L2", "cache_lock_timestamp", "raw_image_access", "runtime_sources", "all_weights"]})
    return values, build, data, model, weights, evaluation_started


def evaluate():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "8"
    sys.dont_write_bytecode = True
    weight_paths = [r["path"] for r in read_json(ROOT / "WEIGHT_IDENTITY.json")["encoders"].values()]
    guard = EvaluationGuard(weight_paths)
    sys.addaudithook(guard.audit)
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits
    blas_controller = threadpool_limits(limits=8, user_api="blas")
    blas_pools = [pool for pool in threadpool_info() if pool["user_api"] == "blas"]
    require(blas_pools and all(pool["num_threads"] == 8 for pool in blas_pools), "Actual NumPy BLAS thread count must be eight")
    require(not (ROOT / "QUALIFICATION_RESULTS.json").exists(), "Qualification results are write-once")
    require(not any((ROOT / "metrics").rglob("RIDGE_METRICS.json")), "No partial previous evaluation may silently rerun")
    try:
        values, build, data, model, weights, evaluation_started = all_eight_preconditions()
        numerics = {}
        result_records = {}
        solve_times = {}
        for name in ENCODERS:
            start = time.perf_counter()
            train = values[name + "_fold0"]
            valid = values[name + "_fold1"]
            X0 = np.column_stack((train["features"].astype(np.float64), np.ones(COUNT, dtype=np.float64)))
            X1 = np.column_stack((valid["features"].astype(np.float64), np.ones(COUNT, dtype=np.float64)))
            Y0 = np.eye(CLASSES, dtype=np.float64)[train["labels"]]
            W = np.linalg.solve(X0.T @ X0 + np.eye(DIMS[name] + 1, dtype=np.float64), X0.T @ Y0)
            s0 = X0 @ W
            s1 = X1 @ W
            require(np.isfinite(W).all() and np.isfinite(s0).all() and np.isfinite(s1).all(), "Finite ridge numerics")
            p0 = np.argmax(s0, axis=1).astype("<i8")
            p1 = np.argmax(s1, axis=1).astype("<i8")
            correct0 = int(np.count_nonzero(p0 == train["labels"]))
            correct1 = int(np.count_nonzero(p1 == valid["labels"]))
            cm = np.zeros((CLASSES, CLASSES), dtype="<i8")
            np.add.at(cm, (valid["labels"], p1), 1)
            solve_times[name] = time.perf_counter() - start
            numerics[name] = {"W": W, "fold0_scores": s0, "fold1_scores": s1,
                              "fold0_predictions": p0, "fold1_predictions": p1, "fold1_confusion": cm}
            result_records[name] = {"fold0_correct": correct0, "fold1_correct": correct1, "count": COUNT,
                                    "fold0_accuracy": correct0 / COUNT, "fold1_accuracy": correct1 / COUNT}
        # All four exact outcomes exist before selection; only integer counts are compared.
        best = CANDIDATES[0]
        for name in CANDIDATES[1:]:
            if result_records[name]["fold1_correct"] > result_records[best]["fold1_correct"]:
                best = name
        qualifies = result_records[best]["fold1_correct"] > result_records["resnet18"]["fold1_correct"]
        selected = best if qualifies else None
        status = "FOOD101_ENCODER_UPGRADE_QUALIFIED" if qualifies else "FOOD101_NO_ENCODER_UPGRADE_QUALIFIED"
        class_names = {index: name for name, index in data["class_to_idx"].items()}
        for name in ENCODERS:
            directory = ROOT / "metrics" / name
            directory.mkdir(parents=True, exist_ok=True)
            numeric_path = directory / "RIDGE_NUMERICS.npz"
            with numeric_path.open("xb") as f:
                np.savez(f, **numerics[name])
            metrics = {"status": "RIDGE_QUALIFICATION_METRICS_FROZEN", "encoder": name, "created_utc": utc(),
                       **result_records[name], "ridge": 1.0, "bias_regularized": True, "precision": "FP64",
                       "formula": "solve(X0.T @ X0 + I, X0.T @ Y0)", "classes": CLASSES,
                       "evaluation_wall_seconds": solve_times[name], "numerics_identity": identity(numeric_path),
                       "feature_build_lock_identity": identity(ROOT / "FEATURE_BUILD_LOCK.json"),
                       "feature_cache_keys": [name + "_fold0", name + "_fold1"]}
            write_json_once(directory / "RIDGE_METRICS.json", metrics)
            cm = numerics[name]["fold1_confusion"]
            with (directory / "FOLD1_CONFUSION.csv").open("x", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["class_id", "class_name"] + ["pred_" + str(i) for i in range(CLASSES)])
                for i in range(CLASSES):
                    writer.writerow([i, class_names[i]] + cm[i].tolist())
            with (directory / "FOLD1_PER_CLASS.csv").open("x", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["class_id", "class_name", "count", "correct", "accuracy"])
                for i in range(CLASSES):
                    writer.writerow([i, class_names[i], int(cm[i].sum()), int(cm[i, i]), int(cm[i, i]) / int(cm[i].sum())])
        completed = utc()
        results = {"status": status, "encoders": result_records, "best_candidate": best, "selected_encoder": selected,
                   "feature_build_lock_identity": identity(ROOT / "FEATURE_BUILD_LOCK.json"),
                   "evaluation_started_utc": evaluation_started, "evaluation_completed_utc": completed,
                   "selection_rule": "Integer correct counts / identical denominator 4242; exact candidate tie priority ConvNeXt > Swin > ViT; strict candidate > ResNet18",
                   "candidate_tie_priority": list(CANDIDATES), "ridge": 1.0, "classes": CLASSES}
        write_json_once(ROOT / "QUALIFICATION_RESULTS.json", results)
        if selected is not None:
            frozen = model["candidates"][selected]
            write_json_once(ROOT / "SELECTED_ENCODER_LOCK.json",
                            {"status": "SELECTED_ENCODER_IDENTITY_FROZEN", "created_utc": utc(), "encoder": selected,
                             "qualification_status": status, "weights_enum": frozen["weights_enum"],
                             "weight_identity": weights["encoders"][selected], "source": frozen["source"],
                             "hook_definition": frozen["feature_hook"], "feature_dimension": DIMS[selected],
                             "transform": build["caches"][selected + "_fold0"]["transform"],
                             "normalization": "per-row FP32 L2", "fold1_correct": result_records[selected]["fold1_correct"],
                             "old_fold1_correct": result_records["resnet18"]["fold1_correct"], "count": COUNT,
                             "qualification_results_identity": identity(ROOT / "QUALIFICATION_RESULTS.json"),
                             "feature_build_lock_identity": identity(ROOT / "FEATURE_BUILD_LOCK.json")})
        costs = {name: {"fold0_encoding_wall_seconds": build["caches"][name + "_fold0"]["encoding_wall_seconds"],
                        "fold1_encoding_wall_seconds": build["caches"][name + "_fold1"]["encoding_wall_seconds"],
                        "fold0_cache_bytes": build["caches"][name + "_fold0"]["bytes"],
                        "fold1_cache_bytes": build["caches"][name + "_fold1"]["bytes"],
                        "cache_bytes": build["caches"][name + "_fold0"]["bytes"] + build["caches"][name + "_fold1"]["bytes"],
                        "ridge_solve_evaluation_seconds": solve_times[name]} for name in ENCODERS}
        write_json_once(ROOT / "COST_ACCOUNTING.json", {"status": "QUALIFICATION_COST_ACCOUNTED", "created_utc": utc(),
                                                       "encoders": costs, "settings": read_json(ROOT / "RUNTIME_GATE.json")["settings"],
                                                       "NumPy_BLAS_pools": blas_pools, "speed_ranking_claim": False})
        require(not guard.attempts, "Forbidden access attempt during evaluation")
        write_json_once(ROOT / "NO_FORBIDDEN_ACCESS_AUDIT.json",
                        {"status": "NO_FORBIDDEN_QUALIFICATION_ACCESS_PASS", "created_utc": utc(),
                         "allowed_image_folds": [0, 1], "encoder_count": 4, "per_encoder_fold_decode_count": COUNT,
                         "total_image_decodes": 4 * 2 * COUNT,
                         "access_logs": {k: rec["access_log"] for k, rec in build["caches"].items()},
                         "build_access_summary_identity": identity(ROOT / "logs" / "FEATURE_BUILD_ACCESS_SUMMARY.json"),
                         "evaluation_preconditions_identity": identity(ROOT / "logs" / "EVALUATION_PRECONDITIONS.json"),
                         "evaluation_forbidden_attempts": guard.attempts, "evaluation_image_opens": 0,
                         "forbidden_fold_image_decodes": {"fold2": 0, "fold3": 0, "fold4": 0, "fold5": 0},
                         "official_train_image_decodes": 0, "procrustes_fit_count": 0, "current_core_execution_count": 0,
                         "revision_result_access_count": 0, "R1_R2_result_access_count": 0,
                         "tuning": False, "auto_confirmation": False,
                         "selection_basis": "clean fold1 exact accuracy after all-eight hash lock",
                         "stop_after_qualification_package": True})
        print(json.dumps(results, ensure_ascii=False, sort_keys=True), flush=True)
    except Exception as exc:
        write_json_once(ROOT / "logs" / ("RIDGE_EVALUATION_FAILURE_" + str(time.time_ns()) + ".json"),
                        {"status": "FOOD101_RIDGE_EVALUATION_STOP", "created_utc": utc(), "failure": str(exc),
                         "traceback": traceback.format_exc(), "forbidden_attempts": guard.attempts})
        raise


if __name__ == "__main__":
    evaluate()
