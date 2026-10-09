#!/usr/bin/env python3
"""Verify that the reviewer-facing artifact contains its required lightweight files."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "LICENSE",
    "LICENSE_DECISION.md",
    "SOURCE_VERIFICATION.md",
    "tables/sample_current_control.csv",
    "REVIEWER_QUICKSTART.md",
    "CITATION.cff",
    "verification/verify_frozen_package_hashes.py",
    "DATA_AND_CODE_AVAILABILITY.md",
    "PAPER_TO_ARTIFACT_MAP.csv",
    "PROVENANCE.md",
    "STATUS.md",
    "protocol/README.md",
    "protocol/cifar_final/STUDY_LOCK.json",
    "protocol/food101_external/STUDY_LOCK.json",
    "tables/cifar_final_primary.csv",
    "tables/food101_external_primary.csv",
    "tables/food101_external_streams.csv",
    "tables/frontend_stress.csv",
    "tables/interaction_decomposition.csv",
    "tables/endpoint_summary.csv",
    "tables/frozen_package_hashes.csv",
    "tables/conditional_bootstrap_food_streams.csv",
    "tables/conditional_bootstrap_primary.csv",
    "tables/correction_size_sensitivity.csv",
    "tables/exact_evidence_coverage.csv",
    "code/README.md",
    "code/frontends/procrustes_alignment.py",
    "code/frontends/check_procrustes.py",
    "code/frontends/fastfill_interface.py",
    "code/frontends/frontend_interface.py",
    "code/frontends/check_fastfill_interface.py",
    "code/analyses/analyze_saved_scores.py",
    "environment/README.md",
    "third_party/ASSET_INSTRUCTIONS.md",
    "verification/run_all.py",
    "verification/verify_inventory.py",
    "verification/verify_release_tables.py",
    "verification/verify_protocol_summaries.py",
    "verification/verify_public_hygiene.py",
    "verification/verify_strengthening_results.py",
]

def main():
    missing = [rel for rel in REQUIRED if not (ROOT / rel).is_file()]
    if missing:
        for rel in missing:
            print(f"MISSING {rel}")
        raise SystemExit(1)
    print(f"PASS: artifact inventory contains {len(REQUIRED)} required lightweight files.")

if __name__ == "__main__":
    main()
