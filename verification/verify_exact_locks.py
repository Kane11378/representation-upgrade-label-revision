#!/usr/bin/env python3
"""Verify accepted row locks, provenance, and common evidence without models.

The ordered identity pins below were read directly from the accepted ZIPs.
No seed is regenerated. Food image pins omit only the private ``path`` field;
all remaining fields, row order, labels, versions, and hashes are included.
Full feature/score payloads are external: this checks their frozen receipts,
not their numerical regeneration. Use --source-dir to additionally verify the
original archives and compare every imported byte/projection to its ZIP member.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import struct
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CORE_SHA256 = "0b28ae97c4fcd65eb901938e06a8642a899e5c32208b239da7b4ac64101d6bd7"
ARCHIVE_PINS = {
    "011_alignment_revision_bridge_20261005.zip": {
        "bytes": 514936,
        "sha256": "0d81ac3da03cfa5b4ddc5d21a518baf56b1fe02e1201d3dedb43900b1cda0539"
    },
    "011_fastfill_final_unseen_protocol_freeze_01_20261006.zip": {
        "bytes": 1490420,
        "sha256": "744ebb41dccaa2a99e16c878b01c2c1dd72b4a1c2850f906a294c50f7d77355c"
    },
    "011_fastfill_final_unseen_confirmation_01_20261006.zip": {
        "bytes": 1563622,
        "sha256": "6dac21472d873856899938f57c8972d3f44fa433ec6f1f3db6a8d8a6b779045d"
    },
    "011_food101_cross_domain_protocol_freeze_01_20261006.zip": {
        "bytes": 10652341,
        "sha256": "5730eaf2d5c9a69777b7c8f5b6000b5af1360abfad97a81abf45958d3bc20a3d"
    },
    "011_food101_encoder_qualification_01_20261006.zip": {
        "bytes": 8394333,
        "sha256": "5fd3e8a50f650b72282273de857266627effaeb2aa688e91e45b13a8b64ed991"
    },
    "011_food101_cross_domain_revision_confirmation_01_20261006.zip": {
        "bytes": 15950893,
        "sha256": "8250a4f98bc1a147787b40cc6ac020d07157fa2c75511d0915ce63fc58805808"
    }
}  # Accepted SHA256/bytes records.
# These pins authenticate the maintained public metadata views. Original
# archive and scientific identity pins above and below remain unchanged.
IMPORT_MANIFEST_DIGEST = "7241371d164ed3e701800e1db7e3ec2cee086878d9af2fb0845ec8761c2b90e5"
SOURCE_ARCHIVES_DIGEST = "8a8ed34d1395b9458fb32c2ddd3023d13db6f07db3372adba4f95eb6c280de23"
FOLD_PINS = {
    "0": "66cce0b425864852f2c54c6d0065491eb39b921b867a3b3136544d77f02fabb4",
    "1": "158a497f4b919699c81e6eba47f48888469e68157e665c4e7406ae7bb2efdcf1",
    "2": "aa0acbf7b3ae7a793a25142aec9042275c2ff6be00b2d0a7fcf0e8875dcad00f",
    "3": "645ccb9bc8bf43f1aa39cd6fe7e8689ee474d0194ff51cee983e24d1f5b3cf8d",
    "4": "8bffd18262b3a23995a71cadf4c790cb53552463cc8a9b194705cf2ef71d0c6f",
    "5": "00efea50a41e11f11f97c225c249663599df3316c183e56007dab44b5de8d70d",
}
FOOD_PINS = {
    "A": {
        "stage": {
            "history": "623970744623a2b4db1067e7f9bec3fd0ef63ff39030c62178e01a603953ab79",
            "E": "252b221c13e4a08de01446237109471db76e1cbd90c07683f410f5d93c0a5a3c"
        },
        "evidence": {
            "fit": "4f9802d59adc7608140f92b849ecb3f8dc968352c382e822dffea1c478415a9e",
            "pilot": "4d8144522e0ecd3eac8b2942a99144fe21c74e10c199fd6ab01fc6e2b173b92d"
        },
        "requests": {
            "R1": "b391b9de346a17cb647e78554c25c05b47925fee375cd2edc7b64c082dfff4eb",
            "R2": "ad410ddb336f3a08a6c8869ee9ec368b6c93ff8bc277ab3dc78e225bc6c3cca0"
        },
        "corruption": "f0e125492b85f670618da18b120298b14b499fe3cd1fece911b46e696f5f3d62"
    },
    "B": {
        "stage": {
            "history": "256871a7beb510e41d4ff6cb7f419c4851617fd4666deebd64f4538fad284495",
            "E": "fb22b5c7b6212a1ed00691de8f85b87fb819bc99d557e81e33b1198ac6b66478"
        },
        "evidence": {
            "fit": "3d82893466ddb57bbb0a986531301e285b2c14d62f1b0dcb7172596ee15fbbe6",
            "pilot": "bdca203994f32b144dc93a2bc0fcf5a7f42869976b0cf4ab37a8508d8f4c1022"
        },
        "requests": {
            "R1": "dc7574ab5749350250375c5b556a3e0e330ff87d76fd25c558ccd4fc32179043",
            "R2": "e40805356db64e5dd9ff61d8c2439cc4c2c438221151a14523aa34cd21d0504f"
        },
        "corruption": "3e3d957f3887ed15e537e10b3c64e4fff296b52938c1040282666c28089b05f2"
    },
    "C": {
        "stage": {
            "history": "8f0db0eab5d23e28de42221d5ad1fce2291f9e54e4662e8589dd95d1bc15e2f9",
            "E": "1db14e7a631f804f894f47fa707d0d4868a6ee85028fa00a79bd29cb4c961064"
        },
        "evidence": {
            "fit": "82fbd28e325e36bce230d7cebcb748810e49f4a5307c020e94c00c6508774816",
            "pilot": "0587d1bd64c66909e5c1d139f74f8cd4182fe020e0ac8c1740723b96fcfb125c"
        },
        "requests": {
            "R1": "4353b2d5e332583ae75d367df74877403fc13085865dd66956ca261beebab2ec",
            "R2": "e6f0bda35d605b58bd2cd11e99b6de4d5715405fe6b4c8a40c3540fddadfbb6e"
        },
        "corruption": "45d91f2fa5a86a7ef7249107f09e5bfafeb4fc1a69cd5bb889ad0a0878018ad8"
    }
}  # Ordered original image payloads with only path omitted.
CIFAR_REQUEST_PINS = {
    "R1": "0bf104cac48851010c5e9e4bf391049a2cd1d897737ea876bc11a60dcba16944",
    "R2": "fea448de9b1ddc26bcb9869230dd1629a04bc824ca602abc8997ab31c9396de2",
}
STAGES = ("before_R1", "after_refresh_R1", "after_R1", "before_R2", "after_refresh_R2", "after_R2")
EVIDENCE_ARRAYS = ("proxy", "observed", "known", "certainty", "pilot", "labels", "versions", "global_rows", "image_ids")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return sha(canonical(value))


def int64_digest(values):
    return sha(struct.pack("<" + "q" * len(values), *values))


def image_digest(images):
    return digest([{k: v for k, v in image.items() if k != "path"} for image in images])


def load(root, relative):
    return json.loads((root / relative).read_text(encoding="utf-8-sig"))


def group_rows(group, count, label, full_class_support=True):
    rows, images = group["rows"], group["images"]
    require(len(rows) == len(images) == len(set(rows)) == count, label + " row count/uniqueness")
    require(rows == sorted(rows) == [x["row"] for x in images], label + " exact row/image order")
    counts = Counter(str(x["class_id"]) for x in images)
    if full_class_support:
        require(set(counts) == {str(i) for i in range(101)}, label + " class support")
    require(group["counts"] == {"images": count, "classes": len(counts), "per_class": dict(counts)}, label + " counts")
    require(len({x["image_id"] for x in images}) == count, label + " image uniqueness")
    return set(rows)


def verify_cifar(split, fit, requests, receipts, public, reserve, method):
    ids = split["selected_ids"]
    require(split["seed"] == split["stream_id"] == fit["seed"] == requests["seed"] == 907001, "CIFAR stream identity")
    require(len(ids) == len(set(ids)) == 6000 and all(0 <= x < 50000 for x in ids), "CIFAR TRAIN namespace")
    require(sorted(ids) == split["canonical_pool_ids"], "CIFAR reserve membership")
    require(int64_digest(sorted(ids)) == "c143c86f777948117dd14e67248ac3cb267ea606c30d76a1ac50ec0b8ca0c20b", "CIFAR exact reserve pin")
    require(int64_digest(ids) == split["selected_ids_sha256"] == "13ba29b606fe0ffc7972394b6842f0310b8598fa45115bc5e42263728b1d2c57", "CIFAR exact ordered split")
    require(sorted(reserve["reserve"]["ids"]) == sorted(ids), "CIFAR source reserve lock")
    for name, start, stop in (("history", 0, 4000), ("arrival_E", 4000, 5000), ("evaluation", 5000, 6000)):
        group = split["groups"][name]
        require(group["rows"] == list(range(start, stop)) and group["global_ids"] == ids[start:stop], "CIFAR " + name + " exact membership")
        require(int64_digest(group["global_ids"]) == group["ids_sha256"], "CIFAR " + name + " identity digest")
    permutation = fit["history_permutation_rows"]
    require(sorted(permutation) == list(range(4000)), "CIFAR evidence permutation identity")
    for name, start in (("fit", 0), ("pilot", 512)):
        group = fit[name]
        require(group["rows"] == permutation[start:start + 512], "CIFAR " + name + " exact ordered selection")
        require(group["global_ids"] == [ids[r] for r in group["rows"]], "CIFAR " + name + " global identity")
        require(int64_digest(group["global_ids"]) == group["ids_sha256"], "CIFAR " + name + " digest")
    initial = set(fit["fit"]["rows"]) | set(fit["pilot"]["rows"])
    require(len(initial) == 1024, "CIFAR fit/pilot disjointness")
    require(all(fit[k] is False for k in ("evidence_selection_uses_labels", "evidence_selection_uses_features", "evidence_selection_uses_sigma", "evidence_selection_uses_requests")), "CIFAR fixed evidence boundary")
    require(public["split"] == split and public["fit_pilot"] == fit and public["method"] == method, "CIFAR confirmation reuse of accepted locks")
    require(method["backend"]["current_core_sha256"] == CORE_SHA256, "CIFAR backend identity")
    require((method["backend"]["alpha"], method["backend"]["ridge"], method["backend"]["shrink"]) == (.25, 1, .1), "CIFAR fixed backend")
    r1, r2 = (requests["requests"][event]["rows"] for event in ("R1", "R2"))
    require(len(r1) == 819 and len(r2) == 1228 and not (set(r1) & set(r2)), "CIFAR request count/disjointness")
    require(r1 + r2[:820] == requests["permuted_wrong_history_rows"], "CIFAR frozen wrong-history request order")
    require(set(r1 + r2[:820]) == set(requests["wrong_history_rows"]), "CIFAR complete wrong-history coverage")
    require(r2[820:] == [4000 + x for x in requests["wrong_arrival_rows"]], "CIFAR frozen arrival requests")
    known = set(initial)
    certainty = set(fit["fit"]["rows"])
    for event in ("R1", "R2"):
        if event == "R2":
            known.update(range(4000, 5000)); certainty.update(range(4000, 5000))
        req = requests["requests"][event]
        rows = req["rows"]
        require(int64_digest(rows) == CIFAR_REQUEST_PINS[event], "CIFAR " + event + " exact request identity")
        require(req["count"] == len(rows) == len(req["entries"]), "CIFAR " + event + " count")
        record = receipts["events"][event]
        fresh = [r for r in rows if r not in known]
        require(record["requested_rows"] == rows and record["fresh_rows"] == fresh, "CIFAR " + event + " acquisition receipt")
        require(record["global_ids"] == [ids[r] for r in rows], "CIFAR " + event + " receipt global identity")
        before = requests["stage_labels"]["before_" + event]
        after = requests["stage_labels"]["after_" + event]
        expected_labels, expected_versions = list(before["labels"]), list(before["versions"])
        for index, (r, entry) in enumerate(zip(rows, req["entries"])):
            require(entry["row"] == r and entry["request_index"] == index and entry["event"] == event and entry["global_id"] == ids[r], "CIFAR ordered request entry")
            require(entry["old_label"] == before["labels"][r] and entry["expected_version"] == before["versions"][r] == 0, "CIFAR prior request state")
            require(entry["old_label"] != entry["new_label"] and entry["resulting_version"] == 1, "CIFAR correction semantics")
            require(entry["target_evidence_known_at_announcement"] == (r in known) and entry["requires_new_target_refresh"] == (r not in known), "CIFAR evidence announcement boundary")
            require(entry["label_release_stage"] == entry["feature_refresh_stage"] == "after_refresh_" + event and entry["label_apply_stage"] == "after_" + event, "CIFAR feature refresh before label apply")
            expected_labels[r], expected_versions[r] = entry["new_label"], 1
        require(after["labels"] == expected_labels and after["versions"] == expected_versions, "CIFAR exact label/version transition")
        for stage in ("before_" + event, "after_refresh_" + event, "after_" + event):
            if stage == "after_refresh_" + event:
                known.update(rows); certainty.update(rows)
            methods = receipts["stages"][stage]
            require(methods["mean"] == methods["paired025"] == methods["sample_current"], "CIFAR shared evidence receipts " + stage)
            rec = methods["mean"]
            require(rec["known_rows"] == sorted(known) and rec["certainty_rows"] == sorted(certainty) and rec["pilot_rows"] == fit["pilot"]["rows"], "CIFAR exact evidence rows " + stage)
            labels = requests["stage_labels"][stage]
            for key in ("labels", "versions"):
                require(int64_digest(labels[key]) == labels[key + "_sha256"] == rec[key]["c_order_raw_sha256"], "CIFAR shared " + key + " identity " + stage)
    require(receipts["final_candidate_rows"] == sorted(known) and receipts["final_candidate_global_ids"] == [ids[r] for r in sorted(known)], "CIFAR final evidence identity")
    require(not (known & set(range(5000, 6000))) and receipts["eval_rows_given_to_candidate"] is False, "CIFAR evaluator isolation")
    require(requests["evaluation_truth"]["global_ids"] == ids[5000:], "CIFAR evaluator truth row identity")


def verify_folds(fold_lock):
    require(set(fold_lock["folds"]) == set(FOLD_PINS), "Food six frozen folds")
    all_rows, all_ids = set(), set()
    byrow = {}
    for number, group in fold_lock["folds"].items():
        rows = group_rows(group, 4242 if int(number) < 4 else 4141, "Food fold " + number)
        ids = {x["image_id"] for x in group["images"]}
        require(not rows & all_rows and not ids & all_ids, "Food fold isolation " + number)
        require(image_digest(group["images"]) == FOLD_PINS[number], "Food exact fold row/image payload " + number)
        require(set(group["counts"]["per_class"].values()) == {42 if int(number) < 4 else 41}, "Food per-class fold size")
        for image in group["images"]:
            byrow[image["row"]] = image
            require(fold_lock["image_assignment"][image["image_id"]] == int(number), "Food image fold assignment")
        all_rows.update(rows); all_ids.update(ids)
    require(all_rows == set(range(25250)) and len(all_ids) == 25250, "Food exact clean-pool coverage")
    require(fold_lock["split_selection_uses_models_or_performance"] is False, "Food fixed fold selection")
    return byrow


def verify_qualification(fold_lock, qualification, deployment, isolation, selected):
    require(set(qualification["folds"]) == {"fold0", "fold1"} and qualification["permitted_performance_folds"] == [0, 1], "Food qualification permitted folds")
    require(set(deployment["folds"]) == {"fold2", "fold3", "fold4", "fold5"}, "Food confirmation deployment folds")
    for i in range(6):
        record = qualification if i < 2 else deployment
        require(record["folds"]["fold" + str(i)] == fold_lock["folds"][str(i)], "Food accepted qualification/deployment fold identity " + str(i))
    require(qualification["folds2_3_4_5_image_payloads_accessed"] is False and qualification["official_train_split_used"] is False, "Food qualification access boundary")
    require(isolation["allowed_image_folds"] == [0, 1] and all(v == 0 for v in isolation["forbidden_fold_image_decodes"].values()), "Food qualification isolation receipts")
    require(isolation["revision_result_access_count"] == isolation["procrustes_fit_count"] == isolation["current_core_execution_count"] == 0, "Food qualification development isolation")
    require(selected["encoder"] == "swin_t" and selected["fold1_correct"] == 2736 and selected["old_fold1_correct"] == 1958 and selected["count"] == 4242, "Food accepted encoder selection")


def verify_food_stream(stream, source_fold, fold_lock, stage, evidence, corruption, requests, receipts, timeline):
    require(stage["stream"] == evidence["stream"] == corruption["stream"] == requests["stream"] == stream, "Food stream identity " + stream)
    history_count = 3333 if stream == "C" else 3434
    history = group_rows(stage["groups"]["history"], history_count, "Food " + stream + " history")
    arrival = group_rows(stage["groups"]["E"], 808, "Food " + stream + " arrival")
    require(not history & arrival and history | arrival == set(fold_lock["folds"][str(source_fold)]["rows"]), "Food exact source partition " + stream)
    for key, data in (("stage", stage["groups"]), ("evidence", evidence["groups"]), ("requests", requests["requests"])):
        for name, group in data.items():
            require(image_digest(group["images"]) == FOOD_PINS[stream][key][name], "Food exact " + key + " payload " + stream + "/" + name)
    require(image_digest(corruption["images"]) == FOOD_PINS[stream]["corruption"], "Food exact corruption payload " + stream)
    fit, pilot = (group_rows(evidence["groups"][name], 512, "Food " + stream + " " + name, full_class_support=False) for name in ("fit", "pilot"))
    require(not fit & pilot and fit | pilot <= history and evidence["label_feature_or_performance_based_selection"] is False, "Food fit/pilot isolation " + stream)
    corrupted = {x["row"]: x for x in corruption["images"] if x["corrupted"]}
    all_corruption = {x["row"]: x for x in corruption["images"]}
    require(set(all_corruption) == history | arrival and len(corrupted.keys() & history) == 1313 and len(corrupted.keys() & arrival) == 303, "Food corruption coverage " + stream)
    for image in corruption["images"]:
        require(image["stage"] == ("history" if image["row"] in history else "E") and image["corrupted"] == (image["historical_label"] != image["class_id"]) and image["label_version"] == 0, "Food corruption label/stage semantics " + stream)
    request_rows = []
    for event, count in (("R1", 606), ("R2", 1010)):
        records = requests["requests"][event]["images"]
        require(len(records) == requests["requests"][event]["count"] == count, "Food request count " + stream + event)
        for image in records:
            old = corrupted[image["row"]]
            require(image["image_id"] == old["image_id"] and image["old_label"] == old["historical_label"] and image["new_label"] == old["class_id"], "Food request exact corruption correction " + stream + event)
            require(image["request"] == event and image["expected_label_version"] == 0 and image["new_label_version"] == 1, "Food request version semantics " + stream + event)
        request_rows.extend(image["row"] for image in records)
    require(len(set(request_rows)) == 1616 and set(request_rows) == set(corrupted), "Food complete disjoint corrections " + stream)
    require(receipts["shared_methods"] == ["mean", "paired025", "sample_current"] and receipts["fold5_in_candidate_evidence"] is False, "Food common-method evidence boundary " + stream)
    require(set(receipts["initial"]["global_rows"]) == fit | pilot and receipts["initial"]["unique_rows"] == 1024, "Food initial evidence identity " + stream)
    known, certainty = fit | pilot, set(fit)
    for event in ("R1", "R2"):
        if event == "R2":
            known.update(arrival); certainty.update(arrival)
        entries = requests["requests"][event]["images"]
        rows = [x["row"] for x in entries]
        rec = receipts["events"][event]
        require(rec["announced_global_rows"] == rows and rec["announced_image_ids"] == [x["image_id"] for x in entries], "Food announced request receipt " + stream + event)
        require(rec["fresh_global_rows"] == [r for r in rows if r not in known] and rec["already_known_global_rows"] == [r for r in rows if r in known], "Food fresh evidence receipt " + stream + event)
        require(rec["refresh_before_label_release"] is True and rec["authoritative"] is True, "Food authoritative refresh boundary " + stream + event)
        for name in ("before_" + event, "after_refresh_" + event, "after_" + event):
            if name == "after_refresh_" + event:
                known.update(rows); certainty.update(rows)
            stage_receipt, state = receipts["stages"][name], timeline["stages"][name]
            require(stage_receipt["mean_receipt_sha256"] == stage_receipt["paired025_receipt_sha256"] == stage_receipt["sample_current_receipt_sha256"] == stage_receipt["shared_receipt_sha256"], "Food shared evidence receipts " + stream + name)
            arrays = stage_receipt["file_identity"]["arrays"]
            require(digest({k: arrays[k] for k in EVIDENCE_ARRAYS}) == stage_receipt["shared_receipt_sha256"], "Food common receipt semantic hash " + stream + name)
            for key in ("global_rows", "labels", "versions", "pilot", "known", "certainty", "image_ids"):
                dtype = arrays[key]["dtype"]
                if dtype in ("<i8", "int64"):
                    raw = struct.pack("<" + "q" * len(state[key]), *state[key])
                elif dtype == "|b1":
                    raw = bytes(int(x) for x in state[key])
                elif dtype == "<U64":
                    raw = b"".join(x.encode("utf-32-le").ljust(64 * 4, b"\0") for x in state[key])
                else:
                    raise ValueError("Unexpected discrete array dtype: " + dtype)
                require(sha(raw) == arrays[key]["c_order_raw_sha256"] and arrays[key]["shape"] == [len(state[key])], "Food discrete state receipt " + stream + name + "/" + key)
            require({r for r, flag in zip(state["global_rows"], state["known"]) if flag} == known and {r for r, flag in zip(state["global_rows"], state["certainty"]) if flag} == certainty, "Food known/certainty exact rows " + stream + name)
            require(stage_receipt["unique_candidate_target_global_rows"] == sorted(known), "Food candidate receipt exact rows " + stream + name)
        before, refresh, after = (timeline["stages"][x + event] for x in ("before_", "after_refresh_", "after_"))
        require(before["labels"] == refresh["labels"] and before["versions"] == refresh["versions"], "Food labels released after refresh " + stream + event)
        expected_labels, expected_versions = list(before["labels"]), list(before["versions"])
        index = {r: i for i, r in enumerate(before["global_rows"])}
        for image in entries:
            i = index[image["row"]]
            require(expected_labels[i] == image["old_label"] and expected_versions[i] == 0, "Food prior label/version " + stream + event)
            expected_labels[i], expected_versions[i] = image["new_label"], 1
        require(after["labels"] == expected_labels and after["versions"] == expected_versions, "Food exact label/version transition " + stream + event)
    require(receipts["candidate_unique_final_rows"] == sorted(known), "Food final exact target evidence " + stream)


def verify_imports(root, source_dir=None, alignment_archive=None):
    manifest = load(root, "release/IMPORTED_FILES.json")
    require(digest(manifest) == IMPORT_MANIFEST_DIGEST, "Public import-ledger identity")
    imported = manifest["files"]
    sources = load(root, "release/SOURCE_ARCHIVES.json")
    require(digest(sources) == SOURCE_ARCHIVES_DIGEST, "Public source-ledger identity")
    require(sources["all_preimport_checks"] == "PASS" and sources["accepted_current_core_sha256"] == CORE_SHA256, "Accepted source preimport gate")
    # Source report is checked against external accepted identity pins below.
    archives = sources.get("archives", sources.get("sources", []))
    require(len(archives) == len(ARCHIVE_PINS), "Accepted source archive inventory")
    seen = set()
    for record in archives:
        name = record["observed_filename"]
        require(name in ARCHIVE_PINS, "Unexpected accepted archive")
        expected = ARCHIVE_PINS[name]
        require(record["bytes"] == expected["bytes"] and record["sha256"] == expected["sha256"], "Accepted frozen package identity " + name)
        require(record["zip_crc"] == record["manifest_exact_coverage"] == record["manifest_bytes_sha256"] == "PASS", "Accepted archive manifest/CRC receipt " + name)
        seen.add(name)
    require(seen == set(ARCHIVE_PINS), "Missing accepted package identity")
    require(len({r["destination"] for r in imported}) == len(imported), "Unique imported member destinations")
    opened = {}
    try:
        for rec in imported:
            destination = rec["destination"]
            target = (root / destination).resolve()
            require(target.is_relative_to(root.resolve()), "Import path stays within artifact")
            raw = target.read_bytes()
            require(len(raw) == rec["bytes"] and sha(raw) == rec["sha256"], "Imported bytes identity " + destination)
            if destination.endswith("current_core.py"):
                require(sha(raw) == CORE_SHA256 and rec["transform"] == "exact_bytes", "Accepted current_core.py identity")
            if source_dir:
                name = rec["archive"]
                if name not in opened:
                    matches = [alignment_archive] if name == "011_alignment_revision_bridge_20261005.zip" and alignment_archive else list(source_dir.rglob(name))
                    require(len(matches) == 1, "Exactly one requested original archive " + name)
                    require(matches[0].name == name, "Original archive basename " + name)
                    archive_raw = matches[0].read_bytes()
                    require(len(archive_raw) == ARCHIVE_PINS[name]["bytes"] and sha(archive_raw) == ARCHIVE_PINS[name]["sha256"], "Original archive SHA256/bytes " + name)
                    archive = zipfile.ZipFile(matches[0])
                    require(archive.testzip() is None, "Original archive CRC " + name)
                    record = next(x for x in archives if x["observed_filename"] == name)
                    members = archive.namelist()
                    require(len(members) == record["members"] == len(set(members)), "Original archive member count/uniqueness " + name)
                    member_name = record["manifest_member"]
                    manifest_raw = archive.read(member_name)
                    require(sha(manifest_raw) == record["manifest_sha256"] and len(manifest_raw) == record["manifest_bytes"], "Original internal manifest identity " + name)
                    members_lock = json.loads(manifest_raw)["files"]
                    member_records = {x["path"]: x for x in members_lock} if isinstance(members_lock, list) else members_lock
                    prefix = member_name.rsplit("/", 1)[0] + "/" if "/" in member_name else ""
                    require(set(members) == {prefix + p for p in member_records} | {member_name}, "Original internal manifest complete coverage " + name)
                    for path, identity in member_records.items():
                        value = archive.read(prefix + path)
                        require(len(value) == identity["bytes"] and sha(value) == identity["sha256"], "Original internal manifest member identity " + path)
                    opened[name] = archive
                member_raw = opened[name].read(rec["member"])
                require(len(member_raw) == rec["source_bytes"] and sha(member_raw) == rec["source_sha256"], "Original frozen member " + destination)
                if rec["transform"] == "exact_bytes":
                    require(raw == member_raw, "Exact original member bytes " + destination)
                else:
                    original = json.loads(member_raw)
                    projected = json.loads(raw)
                    for pointer in rec["removed_json_pointers"]:
                        pieces = [p.replace("~1", "/").replace("~0", "~") for p in pointer.lstrip("/").split("/")]
                        node = original
                        for piece in pieces[:-1]:
                            node = node[int(piece)] if isinstance(node, list) else node[piece]
                        removed = node.pop(pieces[-1])
                        require(isinstance(removed, str) and (re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", removed) or removed.startswith("/")), "Privacy-only projection " + destination)
                    require(original == projected, "Projection preserves every nonpath scientific value " + destination)
    finally:
        for archive in opened.values():
            archive.close()
    return len(imported)


def read_inputs(root):
    cifar = "protocol/cifar_final/accepted/"
    food = "protocol/food101_external/accepted/"
    c = [load(root, cifar + name + ".json") for name in ("FINAL_STREAM_SPLIT_LOCK", "FIT_PILOT_LOCK", "REQUESTS_LOCK", "ACCESS_RECEIPTS", "PUBLIC_PROTOCOL", "RESERVE_IDENTITY", "METHOD_LOCK")]
    fold = load(root, food + "protocol_freeze/SIX_FOLD_LOCK.json")
    qualification = [load(root, food + path + ".json") for path in ("encoder_qualification/QUALIFICATION_DATA_LOCK", "confirmation/DEPLOYMENT_DATA_LOCK", "encoder_qualification/NO_FORBIDDEN_ACCESS_AUDIT", "encoder_qualification/SELECTED_ENCODER_LOCK")]
    shared = load(root, food + "confirmation/ACCESS_RECEIPTS.json")
    streams = {}
    for stream in "ABC":
        streams[stream] = [load(root, food + "protocol_freeze/" + folder + "/" + stream + ".json") for folder in ("STREAM_STAGE_LOCKS", "EVIDENCE_LOCKS", "CORRUPTION_LOCKS", "REQUEST_LOCKS")]
        streams[stream] += [shared["streams"][stream], load(root, food + "confirmation/states/" + stream + "/STATE_TIMELINE.json")]
    return c, fold, qualification, shared, streams


def negative_checks(c, fold, qualification, streams):
    checks = []

    def rejects(name, action):
        try:
            action()
        except (ValueError, KeyError, IndexError):
            checks.append(name)
        else:
            raise ValueError("Tampering was accepted: " + name)

    bad = deepcopy(c[0]); bad["selected_ids"][0], bad["selected_ids"][1] = bad["selected_ids"][1], bad["selected_ids"][0]
    rejects("CIFAR same-count split reorder", lambda: verify_cifar(bad, *c[1:]))
    badfit = deepcopy(c[1]); badfit["pilot"] = deepcopy(badfit["fit"])
    rejects("CIFAR fit/pilot overlap", lambda: verify_cifar(c[0], badfit, *c[2:]))
    badrequest = deepcopy(c[2]); badrequest["requests"]["R1"]["entries"][0]["new_label"] = badrequest["requests"]["R1"]["entries"][0]["old_label"]
    rejects("CIFAR unrepaired requested label", lambda: verify_cifar(c[0], c[1], badrequest, *c[3:]))
    badreceipt = deepcopy(c[3]); badreceipt["stages"]["before_R1"]["paired025"]["proxy"]["c_order_raw_sha256"] = "0" * 64
    rejects("CIFAR unequal shared feature receipt", lambda: verify_cifar(*c[:3], badreceipt, *c[4:]))
    badfold = deepcopy(fold); a, b = badfold["folds"]["0"], badfold["folds"]["2"]
    a["images"][0], b["images"][0] = b["images"][0], a["images"][0]
    rejects("Food same-count fold identity swap", lambda: verify_folds(badfold))
    badqualification = deepcopy(qualification); badqualification[0]["folds"]["fold1"] = deepcopy(fold["folds"]["2"])
    rejects("Food qualification using a deployment fold", lambda: verify_qualification(fold, *badqualification))
    badstream = deepcopy(streams["A"]); badstream[3]["requests"]["R1"]["images"][0]["new_label"] = 100
    rejects("Food altered correction truth", lambda: verify_food_stream("A", 2, fold, *badstream))
    badstream = deepcopy(streams["A"]); badstream[4]["stages"]["before_R1"]["paired025_receipt_sha256"] = "0" * 64
    rejects("Food unequal shared evidence receipt", lambda: verify_food_stream("A", 2, fold, *badstream))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Curated artifact root")
    parser.add_argument("--source-dir", type=Path, help="Directory containing the six original accepted ZIPs")
    parser.add_argument("--alignment-archive", type=Path, help="Original historical alignment ZIP when held outside --source-dir")
    args = parser.parse_args()
    require(args.alignment_archive is None or args.source_dir is not None, "--alignment-archive requires --source-dir")
    count = verify_imports(args.root, args.source_dir, args.alignment_archive)
    c, fold, qualification, shared, streams = read_inputs(args.root)
    verify_cifar(*c)
    verify_folds(fold)
    verify_qualification(fold, *qualification)
    require(shared["fold0_or_fold1_in_confirmation"] is False and shared["fold5_candidate_evidence"] is False, "Food confirmation fold access boundary")
    for stream, source_fold in zip("ABC", (2, 3, 4)):
        verify_food_stream(stream, source_fold, fold, *streams[stream])
    negatives = negative_checks(c, fold, qualification, streams)
    print(f"PASS: {count} frozen imports; exact CIFAR split/evidence/R1/R2; Food six folds/A-B-C corruption/requests/qualification; common evidence receipts; {len(negatives)} tampering checks rejected.")
    if args.source_dir:
        print("PASS: original ZIP SHA256/bytes/CRC and every imported byte or privacy-only projection compared directly.")


if __name__ == "__main__":
    main()
