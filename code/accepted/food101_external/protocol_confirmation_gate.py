"""Metadata/lineage gates and exact accepted self-check/regression only.

No Food101 image/model/cache/performance access and no split/random generation.
The accepted synthetic self-check keeps its original fixed internal RNG unchanged.
"""
from __future__ import annotations
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
PROTOCOL = WORKSPACE / "food101_cross_domain_protocol_freeze_01"
QUALIFICATION = WORKSPACE / "food101_encoder_qualification_01"
ACCEPTED = WORKSPACE / "procrustes_alignment_bridge_01/resume_after_semantic_identity_01"
PINNED_MAIN = "ad6e32115f7bfbeef48ff46a4d2113b44793f987"
ZIP_EXPECTATIONS = {
    "protocol": (PROTOCOL, "011_food101_cross_domain_protocol_freeze_01_20261006.zip", "MANIFEST.json", 10652341, 81, "5730eaf2d5c9a69777b7c8f5b6000b5af1360abfad97a81abf45958d3bc20a3d"),
    "qualification": (QUALIFICATION, "011_food101_encoder_qualification_01_20261006.zip", "MANIFEST.json", 8394333, 106, "5fd3e8a50f650b72282273de857266627effaeb2aa688e91e45b13a8b64ed991"),
    "accepted": (ACCEPTED, "011_procrustes_alignment_bridge_01_resume_after_semantic_identity_20261005.zip", "DELIVERY_MANIFEST.json", 1375697, 283, "05065ef55c2695bf014846146e664d154a6e4907eb383cdb510c54adf6f209ed"),
}
APPROVED_SOURCE_SHA = {
    "current_core.py": "0b28ae97c4fcd65eb901938e06a8642a899e5c32208b239da7b4ac64101d6bd7",
    "procrustes_alignment.py": "59be9742d62f67aa6b1c3c47e139932fa4686f1916a8cfc8b8c2c52c353bb78f",
    "bridge_resume.py": "fee80f3c06088fdb80c0a98d43c14ea8f83f398fdbf4e9db62c14870cac33f9e",
    "run_checks.py": "cbc6fe602f5332009dfef3db9c06c70d5d4866a403cfd52c813df9d8478c50e2",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def exact_copy(source, destination):
    source, destination = Path(source), Path(destination)
    require(destination.resolve().is_relative_to(ROOT), "New copy escapes confirmation root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = source.read_bytes()
    with destination.open("xb") as stream:
        stream.write(payload)
    result = identity(destination)
    require(result["bytes"] == len(payload) and result["sha256"] == hashlib.sha256(payload).hexdigest(), "Byte copy differs")
    return {**result, "source_path": str(source.resolve()), "exact_byte_copy": True}


def canonical_digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_delivery(kind):
    base, filename, manifest_name, expected_bytes, expected_members, expected_sha = ZIP_EXPECTATIONS[kind]
    archive_path = base / "deliverables" / filename
    record = identity(archive_path)
    require(record["bytes"] == expected_bytes and record["sha256"] == expected_sha, kind + " ZIP pinned identity mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)) == expected_members, kind + " ZIP count/duplicates mismatch")
        require(archive.testzip() is None, kind + " ZIP CRC failure")
        raw_manifest = archive.read(manifest_name)
        require(raw_manifest == (base / manifest_name).read_bytes(), kind + " manifest disk/ZIP identity mismatch")
        manifest = json.loads(raw_manifest)
        original_records = manifest["files"]
        records = original_records if isinstance(original_records, dict) else {item["path"]: {"bytes": item["bytes"], "sha256": item["sha256"]} for item in original_records}
        require(len(records) == expected_members - 1 and set(names) == set(records) | {manifest_name}, kind + " full manifest coverage mismatch")
        for relative, expected in records.items():
            path = (base / relative).resolve()
            require(path.is_relative_to(base.resolve()), "Unsafe manifest relative path")
            payload = archive.read(relative)
            require(len(payload) == expected["bytes"] and hashlib.sha256(payload).hexdigest() == expected["sha256"], kind + " ZIP payload differs: " + relative)
            actual = identity(path)
            require(actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"], kind + " original disk payload differs: " + relative)
    record.update(members=expected_members, crc_pass=True, manifest_identity=identity(base / manifest_name), manifest_record_count=len(records), complete_manifest_disk_and_zip_pass=True)
    print(json.dumps({"event": "FROZEN_DELIVERY_VERIFIED", "kind": kind, "members": expected_members}, sort_keys=True), flush=True)
    return record, records


def verify_group(group, dataset_images):
    images = group["images"]
    require(group["rows"] == [item["row"] for item in images], "Canonical group rows differ")
    require(group["canonical_sha256"] == canonical_digest(images), "Existing group canonical digest mismatch")
    for item in images:
        require(item == dataset_images[item["row"]], "Existing original clean image row identity differs")
    return {item["image_id"] for item in images}


def protocol_metadata():
    dataset = read(PROTOCOL / "DATASET_IDENTITY.json")
    six = read(PROTOCOL / "SIX_FOLD_LOCK.json")
    status = read(PROTOCOL / "PROTOCOL_STATUS.json")
    require(status["status"] == "FOOD101_CROSS_DOMAIN_PROTOCOL_FROZEN_AUDIT_PASS" and status["revision_executed"] is False, "Protocol was not frozen without revision")
    require(dataset["status"] == "FOOD101_TEST_POOL_IDENTITY_PASS" and dataset["official_train_split_used"] is False, "Clean test dataset identity status")
    require(six["status"] == "FROZEN" and six["split_selection_uses_models_or_performance"] is False, "Six-fold immutable metadata status")
    mapping = dataset["class_to_idx"]
    require(len(mapping) == 101 and set(mapping.values()) == set(range(101)), "Exact 101-class mapping")
    groups, all_ids, counts = {}, set(), {}
    qualification_ids = {item["image_id"] for fold in (0, 1) for item in six["folds"][str(fold)]["images"]}
    for fold in (2, 3, 4, 5):
        group = six["folds"][str(fold)]
        ids = verify_group(group, dataset["images"])
        per_class = 42 if fold in (2, 3) else 41
        require(len(ids) == len(group["images"]) == 101 * per_class, "Fold count mismatch")
        require(Counter(item["class_id"] for item in group["images"]) == Counter({i: per_class for i in range(101)}), "Fold class balance mismatch")
        require(not ids.intersection(all_ids | qualification_ids), "Confirmation fold intersection or qualification contamination")
        all_ids.update(ids)
        groups["fold" + str(fold)] = group
        counts["fold" + str(fold)] = {"images": len(ids), "classes": 101, "per_class": per_class}
    stream_counts = {}
    for stream, fold in zip(("A", "B", "C"), (2, 3, 4)):
        stages = read(PROTOCOL / "STREAM_STAGE_LOCKS" / (stream + ".json"))
        evidence = read(PROTOCOL / "EVIDENCE_LOCKS" / (stream + ".json"))
        corruption = read(PROTOCOL / "CORRUPTION_LOCKS" / (stream + ".json"))
        requests = read(PROTOCOL / "REQUEST_LOCKS" / (stream + ".json"))
        for lock in (stages, evidence, corruption, requests):
            require(lock["status"] == "FROZEN" and lock["stream"] == stream and lock["source_fold"] == fold, "Frozen stream source/status differs")
        history = verify_group(stages["groups"]["history"], dataset["images"])
        arrivals = verify_group(stages["groups"]["E"], dataset["images"])
        fit = verify_group(evidence["groups"]["fit"], dataset["images"])
        pilot = verify_group(evidence["groups"]["pilot"], dataset["images"])
        source = {item["image_id"] for item in groups["fold" + str(fold)]["images"]}
        expected_history = 3333 if stream == "C" else 3434
        require(len(history) == expected_history and len(arrivals) == 808 and not history & arrivals and history | arrivals == source, "Frozen history/E partition differs")
        require(len(fit) == len(pilot) == 512 and not fit & pilot and fit | pilot <= history, "Frozen fit/pilot identity differs")
        require(stages["common_evaluator_fold"] == 5 and stages["stream_internal_evaluator"] is False, "Frozen evaluator role differs")
        require(evidence["label_feature_or_performance_based_selection"] is False, "Evidence was outcome selected")
        require(corruption["canonical_sha256"] == canonical_digest(corruption["images"]), "Frozen corruption canonical digest differs")
        by_id = {item["image_id"]: item for item in corruption["images"]}
        require(set(by_id) == source and corruption["corrupted_counts"] == {"history": 1313, "E": 303}, "Frozen corruption counts/identity differs")
        observed_corruption = Counter()
        for item in corruption["images"]:
            original = dataset["images"][item["row"]]
            require(all(item[k] == v for k, v in original.items()), "Corruption record clean source differs")
            stage = "history" if item["image_id"] in history else "E"
            require(item["stage"] == stage and item["label_version"] == 0, "Frozen stage/version differs")
            require(0 <= item["historical_label"] < 101 and item["corrupted"] == (item["historical_label"] != item["class_id"]), "Frozen historical-label semantics differs")
            if item["corrupted"]:
                observed_corruption[stage] += 1
        require(dict(observed_corruption) == {"history": 1313, "E": 303}, "Actual frozen corrupted set count differs")
        event_sets = {}
        for event, count in (("R1", 606), ("R2", 1010)):
            request = requests["requests"][event]
            images = request["images"]
            ids = {item["image_id"] for item in images}
            require(len(images) == len(ids) == request["count"] == count, "Frozen request count differs")
            require(request["canonical_sha256"] == canonical_digest(images), "Frozen request canonical digest differs")
            require([item["row"] for item in images] == sorted(item["row"] for item in images), "Frozen request original canonical order differs")
            for item in images:
                original = dataset["images"][item["row"]]
                corrupted = by_id[item["image_id"]]
                require(all(item[k] == v for k, v in original.items()), "Request clean source identity differs")
                require(corrupted["corrupted"] and item["old_label"] == corrupted["historical_label"] and item["new_label"] == corrupted["class_id"], "Request exact old-to-clean label differs")
                require(item["expected_label_version"] == 0 and item["new_label_version"] == 1 and item["request"] == event and item["stage"] == corrupted["stage"], "Request exact frozen version/stage differs")
            event_sets[event] = ids
        require(event_sets["R1"] <= history and not event_sets["R1"] & event_sets["R2"], "R1/R2 identities/stages differ")
        require(event_sets["R1"] | event_sets["R2"] == {key for key, item in by_id.items() if item["corrupted"]}, "Request union differs from exact corrupted set")
        require(requests["count_based_reselection"] is False and requests["label_release_after_feature_refresh"] is True, "Frozen request refresh/release semantics differs")
        stream_counts[stream] = {"source_fold": fold, "source": len(source), "history": len(history), "E": len(arrivals), "fit": len(fit), "pilot": len(pilot), "corrupted_history": 1313, "corrupted_E": 303, "R1": 606, "R2": 1010}
    return dataset, groups, counts, stream_counts


def qualification_metadata():
    result = read(QUALIFICATION / "QUALIFICATION_RESULTS.json")
    selected = read(QUALIFICATION / "SELECTED_ENCODER_LOCK.json")
    status = read(QUALIFICATION / "QUALIFICATION_STATUS.json")
    audit = read(QUALIFICATION / "audit/INDEPENDENT_QUALIFICATION_AUDIT.json")
    runtime = read(QUALIFICATION / "RUNTIME_GATE.json")
    weights = read(QUALIFICATION / "WEIGHT_IDENTITY.json")
    require(result["status"] == status["status"] == selected["qualification_status"] == "FOOD101_ENCODER_UPGRADE_QUALIFIED", "Qualification status differs")
    require(result["selected_encoder"] == status["selected_encoder"] == selected["encoder"] == "swin_t", "Selected encoder is not frozen Swin")
    priority = ("convnext_tiny", "swin_t", "vit_b_16")
    best = max(priority, key=lambda key: result["encoders"][key]["fold1_correct"])
    require(best == "swin_t" and result["best_candidate"] == best and result["encoders"][best]["fold1_correct"] > result["encoders"]["resnet18"]["fold1_correct"], "Frozen integer-count selection differs")
    require(audit["status"] == "INDEPENDENT_QUALIFICATION_AUDIT_PASS" and audit["failure"] is None and audit["qualification_status"] == result["status"] and audit["selected_encoder"] == best, "Qualification independent audit differs")
    require(runtime["status"] == "PASS" and runtime["runtime_status"] == "FOOD101_QUALIFICATION_RUNTIME_IDENTITY_PASS", "Accepted runtime status differs")
    require(runtime["actual_runtime"]["python"] == platform.python_version() == "3.12.14" and runtime["actual_runtime"]["torch"] == "2.14.1+cpu" and runtime["actual_runtime"]["torchvision"] == "0.29.1+cpu", "Accepted runtime identity differs")
    for record in runtime["sources"]:
        actual = identity(record["path"])
        require(actual["bytes"] == record["bytes"] == record["expected_bytes"] and actual["sha256"] == record["sha256"] == record["expected_sha256"] and record["identity_match"], "Actual static installed qualification source differs")
    require(weights["status"] == "ALL_FOUR_WEIGHT_IDENTITIES_PASS", "Accepted all-four weight identity differs")
    model = read(PROTOCOL / "MODEL_CANDIDATE_LOCK.json")
    selected_weights = {}
    for name, frozen in (("resnet18", model["old_encoder"]), ("swin_t", model["candidates"]["swin_t"])):
        record = weights["encoders"][name]
        actual = identity(record["path"])
        require(actual["bytes"] == record["bytes"] and actual["sha256"] == record["sha256"], "Actual accepted weight payload differs: " + name)
        require(record["published_url"] == frozen["weight_url"] and record["published_sha256_prefix"] == frozen["url_sha256_prefix"] and record["published_prefix_match"] and actual["sha256"].startswith(frozen["url_sha256_prefix"]), "Official accepted weight prefix/URL differs")
        selected_weights[name] = record
    require(selected["weight_identity"] == weights["encoders"]["swin_t"] and selected["source"] == model["candidates"]["swin_t"]["source"] and selected["feature_dimension"] == 768, "Selected Swin source/weights/dimension identity differs")
    return selected, runtime, selected_weights


def validate_allowed_scripts(sources):
    check_path, comparator_path = sources["run_checks.py"], sources["check_regression.py"]
    allowed = {"__future__", "json", "pathlib", "numpy", "procrustes_alignment"}
    for name, path in (("run_checks.py", check_path), ("check_regression.py", comparator_path)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imports.update(node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module)
        if name == "check_regression.py":
            allowed = {"__future__", "argparse", "csv", "datetime", "hashlib", "json", "math", "pathlib", "sys", "zipfile", "numpy"}
        require(imports <= allowed, "Existing self-check/regression imports an unexpected dependency")
        require(not imports & {"torch", "torchvision", "bridge_resume", "current_core"}, "Existing check imports model/runner/core")
        if name == "check_regression.py":
            calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else "" for node in ast.walk(tree) if isinstance(node, ast.Call)}
            require(not calls & {"solve", "fit", "build", "evaluate", "train", "manual_seed", "default_rng"}, "Comparator contains forbidden experiment/RNG call")
    return True


def run_existing_checks(copied_sources, accepted_records):
    environment = os.environ.copy()
    environment.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    self_root = ROOT / "audit/accepted_self_check"
    self_root.mkdir(parents=True, exist_ok=False)
    self_command = [sys.executable, "-B", str(copied_sources["run_checks.py"])]
    started = utc()
    process = subprocess.run(self_command, cwd=self_root, env=environment, text=True, capture_output=True, check=False)
    log = ROOT / "logs/ACCEPTED_PROCRUSTES_SELF_CHECK_CONSOLE.txt"
    with log.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(process.stdout + process.stderr)
    require(process.returncode == 0, "Exact accepted synthetic self-check failed")
    self_result = read(self_root / "RESULTS.json")
    require(self_result["passed"] is True and self_result["study"] == "procrustes_alignment_01", "Accepted mathematical/interface self-check result not PASS")
    self_record = {"status": "PASS", "command": self_command, "cwd": str(self_root), "returncode": process.returncode, "started_utc": started, "completed_utc": utc(), "source_identity": identity(copied_sources["run_checks.py"]), "frontend_source_identity": identity(copied_sources["procrustes_alignment.py"]), "outputs": {"RESULTS.json": identity(self_root / "RESULTS.json"), "console": identity(log)}, "scope": "Exact existing synthetic/interface checks; original fixed internal RNG retained; no Food101/model/features/performance"}
    write(ROOT / "audit/ACCEPTED_SELF_CHECK_EXECUTION.json", self_record)
    prior = read(ACCEPTED / "phase1_comparison/PHASE1_COMPARISON.json")
    require(prior["comparison_status"] == "PASS" and prior["checks_failed"] == 0, "Accepted original old-bridge comparison not PASS")
    old_paths = prior["paths"]
    require(accepted_records["code/check_regression.py"]["sha256"] == sha(copied_sources["check_regression.py"]), "Accepted comparator source copy differs")
    output = ROOT / "audit/old_bridge_regression"
    comparator_command = [sys.executable, "-B", str(copied_sources["check_regression.py"])]
    arguments = {"actual-root": old_paths["actual_root"], "inputs-root": old_paths["inputs_root"], "current-core": str(ROOT / "code/current_core.py"), "reference-root": old_paths["reference_root"], "source-zip": old_paths["source_zip"], "diagnostic-zip": old_paths["diagnostic_zip"], "output-dir": str(output), "contract-report": old_paths["contract_report"], "component-report": old_paths["component_report"], "audit-report": old_paths["audit_report"]}
    for key, value in arguments.items():
        comparator_command.extend(["--" + key, value])
    started = utc()
    process = subprocess.run(comparator_command, cwd=ROOT, env=environment, text=True, capture_output=True, check=False)
    log = ROOT / "logs/ACCEPTED_OLD_BRIDGE_REGRESSION_CONSOLE.txt"
    with log.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(process.stdout + process.stderr)
    require(process.returncode == 0, "Exact existing old-bridge comparison failed")
    comparison = read(output / "PHASE1_COMPARISON.json")
    require(comparison["comparison_status"] == "PASS" and comparison["checks_failed"] == 0 and comparison["external_evidence_complete"] is True, "Existing old-bridge comparison is not complete PASS")
    require(comparison["checks_total"] == prior["checks_total"] == 2340 and comparison["prediction_arrays_expected"] == 72 and comparison["prediction_values_expected"] == 72000, "Existing old-bridge regression check count differs")
    regression_record = {"status": "PASS", "comparison_status": comparison["comparison_status"], "checks_total": comparison["checks_total"], "checks_failed": comparison["checks_failed"], "command": comparator_command, "arguments": arguments, "returncode": process.returncode, "started_utc": started, "completed_utc": utc(), "source_identity": identity(copied_sources["check_regression.py"]), "inputs": comparison["read_files"], "outputs": {p.name: identity(p) for p in sorted(output.iterdir()) if p.is_file()}, "console_identity": identity(log), "scope": comparison["scope"], "new_food101_state_or_performance_computed": False}
    write(ROOT / "audit/ACCEPTED_OLD_BRIDGE_REGRESSION_EXECUTION.json", regression_record)
    return self_record, regression_record


def main():
    started = utc()
    own = read(ROOT / "audit/PROTOCOL_CONFIRMATION_GATE_SOURCE_LOCK.json")
    own_source = identity(__file__)
    require(own["files"]["code/protocol_confirmation_gate.py"] == {key: own_source[key] for key in ("bytes", "sha256")}, "Confirmation gate source lock differs")
    repository = read(ROOT / "provenance/REPOSITORY_LOCK.json")
    require(repository["commit"] == PINNED_MAIN and repository["repository"] == "Kane11378/011", "Current frozen main identity differs")
    task_bytes = (ROOT / "provenance/FOOD101_CROSS_DOMAIN_REVISION_CONFIRMATION_01.md").read_bytes()
    require(len(task_bytes) == repository["task_bytes"] and hashlib.sha1(b"blob " + str(len(task_bytes)).encode() + b"\0" + task_bytes).hexdigest() == repository["task_blob"], "Pinned main task Git blob differs")
    protocol_zip, protocol_records = verify_delivery("protocol")
    qualification_zip, qualification_records = verify_delivery("qualification")
    accepted_zip, accepted_records = verify_delivery("accepted")
    project_review = (ROOT / "provenance/PROCRUSTES_ACCEPTED_REVIEW.md").read_text(encoding="utf-8")
    require(accepted_zip["sha256"] in project_review and "1,375,697" in project_review and "283 members" in project_review, "Accepted package not bound by pinned frozen project review")
    dataset, groups, fold_counts, stream_counts = protocol_metadata()
    selected, runtime, weights = qualification_metadata()
    copies = []
    for relative in protocol_records:
        if relative.endswith(".json") and ("/" not in relative or relative.split("/", 1)[0] in {"STREAM_STAGE_LOCKS", "EVIDENCE_LOCKS", "CORRUPTION_LOCKS", "REQUEST_LOCKS"}):
            copies.append(exact_copy(PROTOCOL / relative, ROOT / "provenance/protocol_freeze" / relative))
        elif relative.startswith("provenance/static_reference_source/"):
            copies.append(exact_copy(PROTOCOL / relative, ROOT / "provenance/protocol_freeze" / relative))
    copies.append(exact_copy(PROTOCOL / "MANIFEST.json", ROOT / "provenance/protocol_freeze/MANIFEST.json"))
    copies.append(exact_copy(PROTOCOL / "provenance/MODEL_METHOD_SOURCE_IDENTITY.json", ROOT / "provenance/protocol_freeze/provenance/MODEL_METHOD_SOURCE_IDENTITY.json"))
    for relative in ("RUNTIME_GATE.json", "WEIGHT_IDENTITY.json", "SELECTED_ENCODER_LOCK.json", "QUALIFICATION_RESULTS.json", "QUALIFICATION_STATUS.json", "FEATURE_BUILD_LOCK.json", "NO_FORBIDDEN_ACCESS_AUDIT.json", "audit/INDEPENDENT_QUALIFICATION_AUDIT.json", "MANIFEST.json"):
        copies.append(exact_copy(QUALIFICATION / relative, ROOT / "provenance/encoder_qualification" / relative))
    method_identity = read(PROTOCOL / "provenance/MODEL_METHOD_SOURCE_IDENTITY.json")
    source_paths = {name: ACCEPTED / ("repository_frontend" if name == "run_checks.py" else "code") / name for name in (*APPROVED_SOURCE_SHA, "check_regression.py")}
    accepted_plan = read(ACCEPTED / "PLAN_LOCK.json")
    copied_paths, source_gate_records = {}, {}
    for name, path in source_paths.items():
        actual = identity(path)
        if name in APPROVED_SOURCE_SHA:
            require(actual["sha256"] == APPROVED_SOURCE_SHA[name], "Exact accepted source identity differs: " + name)
        prefix = "repository_frontend/" if name == "run_checks.py" else "code/"
        require(actual["bytes"] == accepted_records[prefix + name]["bytes"] and actual["sha256"] == accepted_records[prefix + name]["sha256"], "Accepted package source payload differs")
        if name in {"current_core.py", "procrustes_alignment.py", "bridge_resume.py"}:
            frozen_key = "approved_" + name
            frozen = method_identity["files"][frozen_key]
            require(actual["bytes"] == frozen["bytes"] and actual["sha256"] == frozen["sha256"] and path.resolve() == Path(frozen["original_absolute_path"]).resolve(), "Frozen protocol accepted method lineage differs")
        if name in {"procrustes_alignment.py", "run_checks.py"}:
            pinned_path = ROOT / "provenance" / ("CURRENT_MAIN_PROCRUSTES_ALIGNMENT.py" if name == "procrustes_alignment.py" else "CURRENT_MAIN_PROCRUSTES_RUN_CHECKS.py")
            require(path.read_bytes() == pinned_path.read_bytes(), "Current pinned repository exact source differs: " + name)
            require(accepted_plan["repository_source_sha256"]["experiments/procrustes_alignment_01/" + name] == actual["sha256"], "Frozen accepted repository source record differs")
        destination = ROOT / "provenance/accepted_procrustes_source" / name
        copies.append(exact_copy(path, destination))
        copied_paths[name] = destination
        source_gate_records[name] = {**identity(destination), "original_source_path": str(path.resolve()), "frozen_protocol_sha256": actual["sha256"] if name in {"current_core.py", "procrustes_alignment.py", "bridge_resume.py"} else None, "accepted_package_sha256": actual["sha256"], "unchanged_code": True}
    for name in ("current_core.py", "procrustes_alignment.py"):
        copies.append(exact_copy(source_paths[name], ROOT / "code" / name))
        source_gate_records[name]["executable_copy"] = identity(ROOT / "code" / name)
    copies.append(exact_copy(source_paths["bridge_resume.py"], ROOT / "provenance/approved_bridge_resume.py"))
    source_gate_records["bridge_resume.py"]["metric_reference_copy"] = identity(ROOT / "provenance/approved_bridge_resume.py")
    for relative in ("PLAN_LOCK.json", "PREFLIGHT.json", "AFFINE_REGRESSION_GATE_PASS.json", "provenance/REPOSITORY_SOURCE_LOCK.json", "phase1_comparison/PHASE1_COMPARISON.json", "DELIVERY_MANIFEST.json", "deliverables/DELIVERY_RECEIPT.json"):
        copies.append(exact_copy(ACCEPTED / relative, ROOT / "provenance/accepted_procrustes_metadata" / relative))
    validate_allowed_scripts(copied_paths)
    write(ROOT / "DEPLOYMENT_DATA_LOCK.json", {"status": "DEPLOYMENT_DATA_LOCK_PASS", "created_utc": utc(), "folds": groups, "class_to_idx": dataset["class_to_idx"], "classes": dataset["classes"], "stream_counts": stream_counts, "protocol_zip": protocol_zip, "confirmation_fold_ids_only": [2, 3, 4, 5], "qualification_folds_excluded": [0, 1], "evaluator_only_fold": 5, "original_groups_preserved": True, "image_payload_accesses": 0, "splits_or_RNG_generated": False})
    write(ROOT / "WEIGHT_IDENTITY.json", {"status": "ALL_TWO_WEIGHT_IDENTITIES_PASS", "created_utc": utc(), "encoders": weights, "accepted_qualification_weight_lock_identity": identity(QUALIFICATION / "WEIGHT_IDENTITY.json"), "actual_payloads_rehashed": True, "weights_loaded_or_deserialized": False, "model_objects_constructed": 0})
    write(ROOT / "PROTOCOL_GATE.json", {"status": "PASS", "semantic_status": "FROZEN_CONFIRMATION_PROTOCOL_IDENTITY_PASS", "created_utc": utc(), "protocol_zip": protocol_zip, "fold_counts": fold_counts, "stream_counts": stream_counts, "dataset_identity": identity(PROTOCOL / "DATASET_IDENTITY.json"), "six_fold_identity": identity(PROTOCOL / "SIX_FOLD_LOCK.json"), "method_lock_identity": identity(PROTOCOL / "METHOD_LOCK.json"), "deployment_data_lock_identity": identity(ROOT / "DEPLOYMENT_DATA_LOCK.json"), "pinned_main": repository, "source_lock_identity": identity(ROOT / "audit/PROTOCOL_CONFIRMATION_GATE_SOURCE_LOCK.json"), "image_payload_accesses": 0, "image_decodes": 0, "performance_computations": 0, "Food101_feature_or_performance_computations": 0, "resplit_reseed_tuning": False})
    write(ROOT / "QUALIFICATION_GATE.json", {"status": "PASS", "semantic_status": "ACCEPTED_SWINT_QUALIFICATION_IDENTITY_PASS", "created_utc": utc(), "qualification_zip": qualification_zip, "selected_encoder": "swin_t", "qualification_status": "FOOD101_ENCODER_UPGRADE_QUALIFIED", "selected_encoder_lock_identity": identity(QUALIFICATION / "SELECTED_ENCODER_LOCK.json"), "selected_identity": selected, "accepted_runtime": runtime, "current_static_source_identities_reverified": True, "weight_identity": identity(ROOT / "WEIGHT_IDENTITY.json"), "qualification_external_feature_payloads_read": False, "new_accuracy_computed": False})
    self_check, regression = run_existing_checks(copied_paths, accepted_records)
    # Reverify all prior delivery payloads after the only two allowed child programs.
    for kind in ZIP_EXPECTATIONS:
        verify_delivery(kind)
    for record in copies:
        actual = identity(record["path"])
        require(actual["bytes"] == record["bytes"] and actual["sha256"] == record["sha256"], "New exact source/provenance copy changed during accepted checks")
    write(ROOT / "PROCRUSTES_SOURCE_GATE.json", {"status": "PASS", "semantic_status": "EXACT_ACCEPTED_PROCRUSTES_AND_OLD_REGRESSION_PASS", "started_utc": started, "completed_utc": utc(), "accepted_package": accepted_zip, "accepted_project_record_identity": identity(ROOT / "provenance/PROCRUSTES_ACCEPTED_REVIEW.md"), "pinned_main": PINNED_MAIN, "sources": source_gate_records, "self_check": self_check, "regression": regression, "self_check_execution_identity": identity(ROOT / "audit/ACCEPTED_SELF_CHECK_EXECUTION.json"), "regression_execution_identity": identity(ROOT / "audit/ACCEPTED_OLD_BRIDGE_REGRESSION_EXECUTION.json"), "all_original_package_files_unchanged_after_checks": True, "copied_source_bytes_unchanged": True, "provenance_copies": copies, "new_Food101_model_or_frontend_or_performance_executed": False, "accepted_bridge_runner_build_evaluate_called": False, "source_lock_identity": identity(ROOT / "audit/PROTOCOL_CONFIRMATION_GATE_SOURCE_LOCK.json")})
    print(json.dumps({"status": "PASS", "protocol_zip": protocol_zip, "qualification_zip": qualification_zip, "accepted_package": accepted_zip, "stream_counts": stream_counts, "self_check_returncode": self_check["returncode"], "regression_returncode": regression["returncode"], "regression_checks": regression["checks_total"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write(ROOT / "audit" / ("PROTOCOL_CONFIRMATION_GATE_FAILURE_" + str(time.time_ns()) + ".json"), {"status": "CONFIRMATION_INPUT_OR_ACCEPTED_REGRESSION_GATE_STOP", "created_utc": utc(), "failure": str(error), "traceback": traceback.format_exc()})
        raise
