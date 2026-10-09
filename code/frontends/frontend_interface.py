"""Compatibility import for the frozen FastFill interface checks.

The implementation remains in fastfill_interface.py. This module exposes its
public objects under the original import name without changing their behavior.
"""
from __future__ import annotations

if __package__:
    from .fastfill_interface import (
        AcquisitionPlan, assert_same_evidence_receipts, fastfill_loss,
        plan_backfill, split_output, validate_preparation_boundary,
    )
else:
    from fastfill_interface import (
        AcquisitionPlan, assert_same_evidence_receipts, fastfill_loss,
        plan_backfill, split_output, validate_preparation_boundary,
    )

__all__ = [
    "AcquisitionPlan", "assert_same_evidence_receipts", "fastfill_loss",
    "plan_backfill", "split_output", "validate_preparation_boundary",
]
