#!/usr/bin/env python3
"""Verify internal arithmetic of the public protocol summaries.

This checks facts available in accepted review records. Exact row-level
identities from the frozen packages are checked separately by
verify_exact_locks.py.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))

def main():
    c = load("protocol/cifar_final/STUDY_LOCK.json")
    f = load("protocol/food101_external/STUDY_LOCK.json")

    # CIFAR reserve and request arithmetic.
    assert c["stream"]["history"] + c["stream"]["arrival_E"] + c["stream"]["evaluation"] == c["reserve"]["count"] == 6000
    assert c["initial_evidence"]["fit_certainty"] == 512
    assert c["initial_evidence"]["pilot"] == 512
    lr = c["labels_and_requests"]
    assert lr["R1"] == lr["history_noisy_ne_clean"] // 2
    assert lr["R2_remaining_history"] == lr["history_noisy_ne_clean"] - lr["R1"]
    assert lr["R2_total"] == lr["R2_remaining_history"] + lr["R2_arrival"] == 1228
    assert c["backend"]["alpha"] == 0.25 and c["backend"]["ridge"] == 1.0 and c["backend"]["shrink"] == 0.1

    # Food-101 six-fold partition.
    folds = f["folds"]
    fold_counts = [
        folds["fold0_qualification_train"], folds["fold1_qualification_val"],
        folds["fold2_stream_A_source"], folds["fold3_stream_B_source"],
        folds["fold4_stream_C_source"], folds["fold5_common_evaluator"]
    ]
    assert sum(fold_counts) == f["clean_source_pool"]["images"] == 25250

    # Maintenance stream coverage and request coverage.
    source_by_stream = {"A": folds["fold2_stream_A_source"],
                        "B": folds["fold3_stream_B_source"],
                        "C": folds["fold4_stream_C_source"]}
    for name, st in f["streams"].items():
        assert st["history"] + st["arrival_E"] == source_by_stream[name]
        assert st["fit"] == st["pilot"] == 512
        assert st["R1"] + st["R2"] == st["history_corruptions"] + st["arrival_corruptions"]
    assert f["encoder_qualification"]["fold1_selected_new_accuracy_pct"] > f["encoder_qualification"]["fold1_old_accuracy_pct"]
    assert f["backend"]["alpha"] == 0.25 and f["backend"]["ridge"] == 1.0 and f["backend"]["shrink"] == 0.1
    assert f["common_evaluator"]["images"] == folds["fold5_common_evaluator"]

    print("PASS: public protocol summaries are internally consistent. "
          "Exact row-level identities are checked separately by verify_exact_locks.py.")

if __name__ == "__main__":
    main()
