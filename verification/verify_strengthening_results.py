#!/usr/bin/env python3
"""Verify the mechanism-strengthening tables added after the frozen confirmations."""
from __future__ import annotations
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
T=ROOT/"tables"

def read(name):
    with (T/name).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    cov=read("exact_evidence_coverage.csv")
    assert len(cov)==6
    mean=[float(x["mean_L"]) for x in cov]
    paired=[float(x["paired_L"]) for x in cov]
    gaps=[abs(float(x["delta_L_paired_minus_mean"])) for x in cov]
    assert all(mean[i+1] <= mean[i]+1e-12 for i in range(5))
    assert all(paired[i+1] <= paired[i]+1e-12 for i in range(5))
    assert all(gaps[i+1] <= gaps[i]+1e-12 for i in range(5))
    assert mean[-1]==paired[-1]==gaps[-1]==0.0

    req=read("correction_size_sensitivity.csv")
    assert [int(x["request_size"]) for x in req]==[1,8,32,128,819]
    assert all(float(x["paired_L"]) < float(x["mean_L"]) for x in req)

    pri=read("conditional_bootstrap_primary.csv")
    assert len(pri)==4 and all(float(x["delta_L_ci_high"])<0 for x in pri)

    stream=read("conditional_bootstrap_food_streams.csv")
    assert len(stream)==6 and all(float(x["delta_L_ci_high"])<0 for x in stream)

    print("PASS: coverage convergence, request-size response direction, and conditional bootstrap intervals verified.")

if __name__=="__main__":
    main()
