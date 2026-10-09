# Protocol Metadata

This directory separates two levels of protocol evidence.

## Protocol summaries

- `cifar_final/STUDY_LOCK.json`
- `food101_external/STUDY_LOCK.json`

These files contain only facts directly supported by accepted protocol/confirmation review records. They enable lightweight checks of counts, request arithmetic, model-selection separation, and frozen hyperparameters.

They are **not** row-level protocol locks.

The summaries retain their original `provenance.note` fields and scientific
identities. Row-level imports are available under `accepted/` and are verified
by the source and exact-lock checks.

## Frozen row-level identities

The `accepted/` subdirectories contain original accepted CSVs and explicitly
documented privacy projections of path-bearing JSON, sufficient to verify:
- dataset/global IDs;
- split membership;
- fit/pilot membership;
- label versions;
- R1/R2 requested IDs;
- evidence receipts / acquisition identities;
- relevant semantic hashes.

These come only from the verified original delivery ZIPs identified in
`../release/SOURCE_ARCHIVES.json`. Every retained scientific row/value/order is
equal to the original. Original and projected hashes plus removed locator field
pointers are in `../release/IMPORTED_FILES.json`. The projected full JSON bytes
are not represented as original byte identities and do not replace the runner's
original execution metadata. `../verification/verify_exact_locks.py` checks row
identities, disjointness, requests, corruption, qualification isolation and shared
evidence receipts, with an optional direct-original comparison mode.

## Source identity

The accepted lock files are the source of truth for scientific row identities. Regenerating them from seeds or the paper description does not establish their original identity.
