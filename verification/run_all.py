#!/usr/bin/env python3
"""Run all lightweight reviewer verification checks."""
from __future__ import annotations
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True

for script in ["verify_inventory.py", "verify_release_tables.py", "verify_protocol_summaries.py", "verify_frozen_package_hashes.py", "verify_strengthening_results.py", "verify_exact_locks.py", "verify_citation.py", "verify_public_hygiene.py", "verify_release_manifest.py"]:
    print(f"== {script} ==")
    runpy.run_path(str(HERE / script), run_name="__main__")

print("ALL_LIGHTWEIGHT_CHECKS_PASS")
