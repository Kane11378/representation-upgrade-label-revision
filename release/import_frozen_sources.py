#!/usr/bin/env python3
"""Verify all accepted ZIPs before importing the explicit public whitelist.

Accepted Python is copied byte-for-byte. JSON containing private locators is
published as a deterministic privacy projection; it is never described as an
original byte-identical lock. This utility performs no fitting or inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

CORE_SHA256 = "0b28ae97c4fcd65eb901938e06a8642a899e5c32208b239da7b4ac64101d6bd7"
ARCHIVES = [
    ("legacy_alignment", "011_alignment_revision_bridge_20261005(1).zip", 514936, 95,
     "0d81ac3da03cfa5b4ddc5d21a518baf56b1fe02e1201d3dedb43900b1cda0539", "alignment_revision_bridge_20261005/DELIVERY_MANIFEST.json"),
    ("cifar_protocol", "011_fastfill_final_unseen_protocol_freeze_01_20261006.zip", 1490420, 46,
     "744ebb41dccaa2a99e16c878b01c2c1dd72b4a1c2850f906a294c50f7d77355c", "MANIFEST.json"),
    ("cifar_confirmation", "011_fastfill_final_unseen_confirmation_01_20261006.zip", 1563622, 63,
     "6dac21472d873856899938f57c8972d3f44fa433ec6f1f3db6a8d8a6b779045d", "MANIFEST.json"),
    ("food_protocol", "011_food101_cross_domain_protocol_freeze_01_20261006.zip", 10652341, 81,
     "5730eaf2d5c9a69777b7c8f5b6000b5af1360abfad97a81abf45958d3bc20a3d", "MANIFEST.json"),
    ("food_qualification", "011_food101_encoder_qualification_01_20261006.zip", 8394333, 106,
     "5fd3e8a50f650b72282273de857266627effaeb2aa688e91e45b13a8b64ed991", "MANIFEST.json"),
    ("food_confirmation", "011_food101_cross_domain_revision_confirmation_01_20261006.zip", 15950893, 191,
     "8250a4f98bc1a147787b40cc6ac020d07157fa2c75511d0915ce63fc58805808", "MANIFEST.json"),
]
TASK_DIRS = {
    "cifar_protocol": "fastfill_final_unseen_protocol_freeze_01",
    "cifar_confirmation": "fastfill_final_unseen_confirmation_01",
    "food_protocol": "food101_cross_domain_protocol_freeze_01",
    "food_qualification": "food101_encoder_qualification_01",
    "food_confirmation": "food101_cross_domain_revision_confirmation_01",
}
# Boundaries prevent the trailing letter of a public HTTPS URL being treated
# as a Windows drive. Source URL and image-relative IDs remain intact.
LOCAL_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9])[a-z]:[\\/]|"
    r"(?<![A-Za-z0-9:])/(?:home|Users|mnt|tmp|opt|usr|var|root|workspace|data)/"
)
PRIVATE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]{1,100}@[A-Za-z0-9.-]{1,100}\.[A-Za-z]{2,}\b")
TOKEN = re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}")
HISTORICAL_LEDGER_SHA256 = {
    "release/IMPORTED_FILES.json": "80cca7fbfffcad21de41879bf5fbf8d8ecc1ba22bdef13cf9b4b7915c38c1a89",
    "release/SOURCE_ARCHIVES.json": "23ea8b96cd80333dae419b5efd48fa6af4e995e35d010fdab36c9d023efe150a",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def metadata_view(path):
    """Identify the immutable original of this reader-oriented ledger view."""
    source = path.endswith("SOURCE_ARCHIVES.json")
    return {
        "kind": "reader-oriented-source-ledger" if source else "reader-oriented-import-ledger",
        "historical_release": "v1.0.0",
        "historical_path": path,
        "historical_file_sha256": HISTORICAL_LEDGER_SHA256[path],
        "description": (
            "Acceptance-record locators are omitted from this reader-oriented view; frozen source identities and verification receipts are unchanged."
            if source else
            "Exclusion descriptions are adapted for readers; selected files, source identities, and projection receipts are unchanged."
        ),
    }


def is_local_path(value):
    return bool(LOCAL_PATH.search(value)) or value.startswith("/")


def projection(value, pointer="", removed=None):
    """Delete private locator leaf fields; preserve every other value/order.

    A list of locators is rejected rather than changing row-array indexing.
    Private dictionary keys are also rejected rather than losing identities.
    """
    if removed is None:
        removed = []
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if is_local_path(key):
                raise ValueError("Private dictionary key requires separate curation")
            field = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            if isinstance(item, str) and is_local_path(item):
                removed.append(field)
            else:
                result[key] = projection(item, field, removed)
        return result
    if isinstance(value, list):
        if any(isinstance(item, str) and is_local_path(item) for item in value):
            raise ValueError("Locator list requires separate curation")
        return [projection(item, pointer + "/" + str(i), removed) for i, item in enumerate(value)]
    return value


def assert_preserved(original, public, pointer=""):
    """Independently compare every nonlocator field, including row arrays."""
    if isinstance(original, dict):
        expected = {key for key, item in original.items()
                    if not (isinstance(item, str) and is_local_path(item))}
        assert set(public) == expected, pointer
        for key in expected:
            assert_preserved(original[key], public[key], pointer + "/" + key)
    elif isinstance(original, list):
        assert isinstance(public, list) and len(public) == len(original), pointer
        for i, (left, right) in enumerate(zip(original, public)):
            assert_preserved(left, right, pointer + "/" + str(i))
    else:
        assert type(original) is type(public) and original == public, pointer


def safety_check(data):
    text = data.decode("utf-8-sig")
    assert not LOCAL_PATH.search(text), "Private local locator remains"
    assert not PRIVATE_EMAIL.search(text), "Private email remains"
    assert not TOKEN.search(text), "Credential pattern remains"


def selections():
    """Each tuple names a ZIP member and its artifact-relative destination."""
    result = {}
    result["legacy_alignment"] = [
        ("alignment_revision_bridge_20261005/ENVIRONMENT.json", "release/source_environment/legacy_alignment_ENVIRONMENT.json")
    ]
    cp = "protocol/cifar_final/accepted/"
    result["cifar_protocol"] = [(name, cp + name) for name in (
        "FINAL_STREAM_SPLIT_LOCK.json", "FIT_PILOT_LOCK.json", "REQUESTS_LOCK.json",
        "METHOD_LOCK.json", "RESERVE_IDENTITY.json", "SOURCE_PROTOCOL_IDENTITY.json",
        "ALGORITHM_SOURCE_LOCK.json", "provenance/PROTOCOL_SEED_LOCK.json")]
    cc = "code/accepted/cifar_final/"
    result["cifar_confirmation"] = [("code/" + name, cc + name) for name in (
        "current_core.py", "confirmation.py", "common_io.py", "encode_reserve.py",
        "dino_io.py", "dino_s_io.py", "fastfill_proxy.py", "selected_train.py",
        "protocol_gate.py", "independent_confirmation_audit.py")]
    result["cifar_confirmation"] += [(name, cp + name) for name in (
        "PUBLIC_PROTOCOL.json", "ACCESS_RECEIPTS.json", "REQUESTS_ALL.csv",
        "FEATURE_CACHE_LOCK.json", "DINO_IDENTITY_GATE.json", "FASTFILL_PROXY_LOCK.json",
        "BUILD_LOCK.json", "EVALUATION_LOCK.json", "EXECUTION_SOURCE_LOCK.json",
        "EXTERNAL_LARGE_ARTIFACTS.json", "PROTOCOL_GATE.json")]
    fp = "protocol/food101_external/accepted/protocol_freeze/"
    result["food_protocol"] = [(name, fp + name) for name in (
        "SIX_FOLD_LOCK.json", "DATASET_IDENTITY.json", "METHOD_LOCK.json",
        "MODEL_CANDIDATE_LOCK.json", "QUALIFICATION_RULE_LOCK.json",
        "CONVENTIONS_LOCK.json", "CLAIM_LOCK.json", "PROTOCOL_STATUS.json")]
    for group in ("CORRUPTION_LOCKS", "EVIDENCE_LOCKS", "REQUEST_LOCKS", "STREAM_STAGE_LOCKS"):
        result["food_protocol"] += [(f"{group}/{stream}.json", fp + f"{group}/{stream}.json") for stream in "ABC"]
    fq = "protocol/food101_external/accepted/encoder_qualification/"
    result["food_qualification"] = [(name, fq + name) for name in (
        "QUALIFICATION_DATA_LOCK.json", "SELECTED_ENCODER_LOCK.json",
        "QUALIFICATION_RESULTS.json", "QUALIFICATION_STATUS.json", "FEATURE_BUILD_LOCK.json",
        "RUNTIME_GATE.json", "WEIGHT_IDENTITY.json", "ARTIFACT_CONTRACT.json",
        "EXECUTION_SOURCE_LOCK_V2.json", "NO_FORBIDDEN_ACCESS_AUDIT.json")]
    for encoder in ("resnet18", "swin_t", "convnext_tiny", "vit_b_16"):
        for fold in (0, 1):
            name = f"FEATURE_CACHE_LOCKS/{encoder}_fold{fold}.json"
            result["food_qualification"].append((name, fq + name))
    result["food_qualification"] += [("code/" + name, "code/accepted/food101_qualification/" + name)
                                        for name in ("ridge_qualification_v2.py", "protocol_gate.py")]
    fc = "protocol/food101_external/accepted/confirmation/"
    result["food_confirmation"] = [("code/" + name, "code/accepted/food101_external/" + name) for name in (
        "current_core.py", "confirmation_bridge_v2.py", "procrustes_alignment.py", "protocol_confirmation_gate.py")]
    result["food_confirmation"] += [(name, fc + name) for name in (
        "DEPLOYMENT_DATA_LOCK.json", "ACCESS_RECEIPTS.json", "REQUESTS_ALL.csv",
        "REQUEST_MANIFEST_ALL.csv", "RUNTIME_GATE.json", "ENCODER_FEATURE_LOCK.json",
        "WEIGHT_IDENTITY.json", "SCORE_LOCK.json", "BUILD_LOCK.json",
        "QUALIFICATION_GATE.json", "PROTOCOL_GATE.json", "ARTIFACT_CONTRACT.json",
        "EXECUTION_SOURCE_LOCK_V2.json", "CONFIRMATION_STATUS.json", "EXTERNAL_ASSETS.json")]
    for encoder in ("resnet18", "swin_t"):
        for fold in (2, 3, 4, 5):
            name = f"FEATURE_CACHE_LOCKS/{encoder}_fold{fold}.json"
            result["food_confirmation"].append((name, fc + name))
    result["food_confirmation"] += [(f"FRONTEND_LOCKS/{stream}.json", fc + f"FRONTEND_LOCKS/{stream}.json") for stream in "ABC"]
    result["food_confirmation"] += [(f"states/{stream}/STATE_TIMELINE.json", fc + f"states/{stream}/STATE_TIMELINE.json") for stream in "ABC"]
    return result


def exclusion_reason(member):
    if member.endswith("official_transformations.py"):
        return "Apple copyright source: redistribution license not supplied in accepted package; obtain upstream license before inclusion"
    if member.endswith(("encode_confirmation.py", "encode_qualification_v2.py")):
        return "Encoder source embeds absolute filesystem locators; no edited execution code is distributed."
    if member.startswith(("attempts/", "diff/", "logs/", "audit/")):
        return "Not selected for this reproduction artifact; original member identity is retained."
    if member.startswith("provenance/"):
        return "Source snapshot outside the selected artifact scope; original member identity is retained."
    return "Not selected for this reproduction artifact; original member identity is retained."


def verify_archive(path, spec):
    ident, requested, expected_bytes, expected_members, expected_sha, manifest_name = spec
    aliases = [requested]
    if ident == "legacy_alignment":
        aliases.append("011_alignment_revision_bridge_20261005.zip")
    assert path.name in aliases, "Unexpected archive basename"
    package = path.read_bytes()
    assert len(package) == expected_bytes and digest(package) == expected_sha, "Frozen archive identity differs"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert len(names) == expected_members and len(set(names)) == len(names)
        for name in names:
            member = PurePosixPath(name)
            assert not member.is_absolute() and ".." not in member.parts and "\\" not in name
        assert archive.testzip() is None, "ZIP CRC failure"
        manifest_bytes = archive.read(manifest_name)
        manifest = json.loads(manifest_bytes)
        files = manifest["files"]
        records = {item["path"]: item for item in files} if isinstance(files, list) else files
        prefix = manifest_name.rsplit("/", 1)[0] + "/" if "/" in manifest_name else ""
        assert set(names) == {prefix + name for name in records} | {manifest_name}
        payloads = {}
        for relative, identity in records.items():
            member_name = prefix + relative
            data = archive.read(member_name)
            assert len(data) == identity["bytes"] and digest(data) == identity["sha256"], member_name
            payloads[member_name] = data
        if ident in {"cifar_confirmation", "food_confirmation"}:
            assert digest(payloads["code/current_core.py"]) == CORE_SHA256
    record = {
        "id": ident, "requested_filename": requested, "observed_filename": path.name,
        "filename_verification": "exact" if path.name == requested else "historically_accepted_basename_alias",
        "filename_note": "Requested download suffix absent; original historical basename verified by accepted byte identity" if path.name != requested else None,
        "bytes": expected_bytes, "sha256": expected_sha, "members": expected_members,
        "zip_crc": "PASS", "manifest_member": manifest_name,
        "manifest_sha256": digest(manifest_bytes), "manifest_bytes": len(manifest_bytes),
        "manifest_payloads_checked": len(records), "manifest_identity": {key: value for key, value in manifest.items() if key != "files"},
        "manifest_exact_coverage": "PASS", "manifest_bytes_sha256": "PASS",
    }
    return record, payloads, manifest_bytes


def write_once_or_identical(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, "Existing import differs: " + path.name
    else:
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Original delivery parent directory")
    parser.add_argument("--alignment-archive", type=Path, required=True, help="Historical alignment original ZIP")
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    verified = []
    # All six archives pass filename/bytes/hash/CRC/full manifest first. No writes
    # or ZIP extraction occurs while validation remains incomplete.
    for spec in ARCHIVES:
        ident, filename = spec[:2]
        path = args.alignment_archive if ident == "legacy_alignment" else args.source_root / TASK_DIRS[ident] / "deliverables" / filename
        verified.append(verify_archive(path, spec))
    by_id = {spec[0]: item for spec, item in zip(ARCHIVES, verified)}
    snapshots = []
    for destination_id, source_id, prefix, expected_count in (
        ("food_qualification", "food_protocol", "provenance/protocol_freeze/", 7),
        ("food_confirmation", "food_protocol", "provenance/protocol_freeze/", 41),
        ("food_confirmation", "food_qualification", "provenance/encoder_qualification/", 9),
    ):
        source_record, source_payloads, source_manifest = by_id[source_id]
        source_files = dict(source_payloads)
        source_files[source_record["manifest_member"]] = source_manifest
        copied = by_id[destination_id][1]
        checked = 0
        for member, data in copied.items():
            if member.startswith(prefix):
                original_member = member[len(prefix):]
                if original_member in source_files:
                    assert data == source_files[original_member], "Copied accepted source identity differs"
                    checked += 1
        assert checked == expected_count
        snapshots.append({"archive_id": destination_id, "source_archive_id": source_id,
                          "member_prefix": prefix, "members_compared": checked, "exact_identity": "PASS"})
    rows = []
    exclusions = []
    pending_writes = []
    selected = selections()
    for spec, (record, payloads, manifest_data) in zip(ARCHIVES, verified):
        ident = spec[0]
        imports = dict(selected[ident])
        for member, destination in selected[ident]:
            original = payloads[member]
            removed = []
            public = original
            transform = "exact_bytes"
            if member.endswith(".json"):
                value = json.loads(original)
                projected = projection(value, removed=removed)
                assert_preserved(value, projected)
                if removed:
                    public = json_bytes(projected)
                    transform = "privacy_projection"
            safety_check(public)
            path = args.artifact_root / destination
            assert path.resolve().is_relative_to(args.artifact_root.resolve())
            pending_writes.append((path, public))
            rows.append({"archive_id": ident, "archive": record["observed_filename"],
                         "member": member, "destination": destination,
                         "source_bytes": len(original), "source_sha256": digest(original),
                         "bytes": len(public), "sha256": digest(public), "transform": transform,
                         "removed_json_pointers": removed, "nonpath_deep_equality": "PASS"})
        # The source manifest is safe and copied exactly to preserve membership
        # identities; it describes external originals, not projected file bytes.
        destination = "release/source_manifests/" + ident + ".json"
        safety_check(manifest_data)
        pending_writes.append((args.artifact_root / destination, manifest_data))
        rows.append({"archive_id": ident, "archive": record["observed_filename"],
                     "member": record["manifest_member"], "destination": destination,
                     "source_bytes": len(manifest_data), "source_sha256": digest(manifest_data),
                     "bytes": len(manifest_data), "sha256": digest(manifest_data),
                     "transform": "exact_bytes", "removed_json_pointers": [], "nonpath_deep_equality": "PASS"})
        for member, original in payloads.items():
            if member not in imports:
                exclusions.append({"archive_id": ident, "member": member,
                                   "source_bytes": len(original), "source_sha256": digest(original),
                                   "reason": exclusion_reason(member)})
    # Complete all privacy/projection/equality checks before any curated writes.
    for path, data in pending_writes:
        write_once_or_identical(path, data)
    source_record = {"schema": "accepted-source-archives-reader-v2",
                     "metadata_view": metadata_view("release/SOURCE_ARCHIVES.json"), "all_preimport_checks": "PASS",
                     "accepted_current_core_sha256": CORE_SHA256, "archives": [item[0] for item in verified],
                     "copied_source_identities": snapshots}
    import_record = {"schema": "curated-frozen-imports-reader-v2",
                     "metadata_view": metadata_view("release/IMPORTED_FILES.json"), "projection_rule": "Omit only dictionary string fields containing absolute filesystem locators; preserve all other values, arrays and ordering without scientific regeneration",
                     "original_path_bearing_json_published": False, "accepted_execution_code_edited": False,
                     "files": sorted(rows, key=lambda item: item["destination"]), "excluded": exclusions}
    # Regenerate the release-engineering ledgers when the explicit whitelist
    # grows; existing accepted/projection payloads must remain identical above.
    (args.artifact_root / "release/SOURCE_ARCHIVES.json").write_bytes(json_bytes(source_record))
    (args.artifact_root / "release/IMPORTED_FILES.json").write_bytes(json_bytes(import_record))
    exact = sum(row["transform"] == "exact_bytes" for row in rows)
    print(f"PASS: 6 accepted archives fully verified; {exact} exact files, {len(rows) - exact} privacy projections imported; {len(exclusions)} unselected original payloads excluded.")


if __name__ == "__main__":
    main()
