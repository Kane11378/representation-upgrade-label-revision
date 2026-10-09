#!/usr/bin/env python3
"""Validate syntax and uniqueness of frozen accepted-package SHA256 identities."""
from __future__ import annotations
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tables" / "frozen_package_hashes.csv"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_PACKAGES = {
    "011_procrustes_alignment_bridge_01_resume_after_semantic_identity_20261005.zip",
    "011_fastfill_common_evidence_revision_bridge_01_20261006.zip",
    "011_fastfill_bridge_maintenance_decomposition_01_20261006.zip",
    "011_fastfill_final_unseen_confirmation_01_20261006.zip",
    "011_food101_cross_domain_revision_confirmation_01_20261006.zip",
}

def main():
    with PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(EXPECTED_PACKAGES)
    packages = {r["package"] for r in rows}
    hashes = [r["sha256"] for r in rows]
    assert packages == EXPECTED_PACKAGES
    assert all(HEX64.fullmatch(x) for x in hashes)
    assert len(set(hashes)) == len(hashes)
    print("PASS: five frozen accepted-package SHA256 identities are well-formed and unique.")

if __name__ == "__main__":
    main()
