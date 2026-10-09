#!/usr/bin/env python3
"""Lightweight verification of the machine-readable paper result tables.

Standard-library only. This verifies published arithmetic and key directional
claims. It is not a substitute for the full frozen experimental rerun.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "tables"

def read(name):
    with (TABLES / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def close(a, b, tol=5e-4):
    if not math.isclose(float(a), float(b), abs_tol=tol, rel_tol=0.0):
        raise AssertionError(f"{a} != {b} within {tol}")

def pct_reduction(mean, paired):
    return (float(mean) - float(paired)) / float(mean) * 100.0

def pct_change(mean, paired):
    return (float(paired) - float(mean)) / float(mean) * 100.0

def verify_primary(name):
    rows = read(name)
    for r in rows:
        close(pct_reduction(r["mean_response_rel"], r["paired_response_rel"]),
              r["response_reduction_pct"])
        close(pct_change(r["mean_action_rel"], r["paired_action_rel"]),
              r["action_change_pct"])

def main():
    verify_primary("cifar_final_primary.csv")
    verify_primary("food101_external_primary.csv")

    streams = read("food101_external_streams.csv")
    assert len(streams) == 6
    assert all(float(r["paired_response_rel"]) < float(r["mean_response_rel"])
               for r in streams), "Food-101 response direction is not 6/6"

    decomp = read("interaction_decomposition.csv")
    for r in decomp:
        lhs = (float(r["delta_refresh_energy"])
               + float(r["delta_revision_response_energy"])
               + float(r["delta_cross_term"]))
        close(lhs, r["delta_composed_action_energy"], tol=2e-9)

    endpoints = read("endpoint_summary.csv")
    assert {r["dataset"] for r in endpoints} == {"CIFAR-100N", "Food-101"}
    assert all(float(r["paired_score_rms"]) < float(r["mean_score_rms"])
               for r in endpoints)

    print("PASS: primary arithmetic, 6/6 Food-101 response direction, "
          "interaction identity, and endpoint score-RMS direction verified.")

if __name__ == "__main__":
    main()
