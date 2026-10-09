# Source Archive and Artifact Verification

This document records source-package verification and integrity checks for
the reproducibility artifact. These checks do not constitute a new scientific rerun.

## Source package identities

| Requested original ZIP | Bytes | SHA256 |
|---|---:|---|
| `011_alignment_revision_bridge_20261005(1).zip` | 514936 | `0d81ac3da03cfa5b4ddc5d21a518baf56b1fe02e1201d3dedb43900b1cda0539` |
| `011_fastfill_final_unseen_protocol_freeze_01_20261006.zip` | 1490420 | `744ebb41dccaa2a99e16c878b01c2c1dd72b4a1c2850f906a294c50f7d77355c` |
| `011_fastfill_final_unseen_confirmation_01_20261006.zip` | 1563622 | `6dac21472d873856899938f57c8972d3f44fa433ec6f1f3db6a8d8a6b779045d` |
| `011_food101_cross_domain_protocol_freeze_01_20261006.zip` | 10652341 | `5730eaf2d5c9a69777b7c8f5b6000b5af1360abfad97a81abf45958d3bc20a3d` |
| `011_food101_encoder_qualification_01_20261006.zip` | 8394333 | `5fd3e8a50f650b72282273de857266627effaeb2aa688e91e45b13a8b64ed991` |
| `011_food101_cross_domain_revision_confirmation_01_20261006.zip` | 15950893 | `8250a4f98bc1a147787b40cc6ac020d07157fa2c75511d0915ce63fc58805808` |

The alignment file was available under the historical basename without `(1)`.
It matches every accepted byte/hash/CRC/manifest identity; this filename alias is
recorded explicitly, without claiming the requested spelling was present.
All six original ZIPs were validated before import: bytes, SHA256, CRC, manifest
identity and complete coverage of 576 nonmanifest payloads. Frozen Food protocol
and qualification copies embedded in later packages were also compared exactly.
See `release/SOURCE_ARCHIVES.json`.

## Imported and excluded files

Imported 109 selected original members: 43 byte-exact files and 66 deterministic
privacy projections. Both accepted core copies retain the required SHA256
`0b28ae97c4fcd65eb901938e06a8642a899e5c32208b239da7b4ac64101d6bd7`.
CIFAR `confirmation.py`, Food `confirmation_bridge_v2.py`, reusable helpers,
row CSVs, split/fit-pilot/evidence/request/corruption/qualification locks, safe
source manifests and runtime/cache/weight identities are included.

Original path-bearing JSON is excluded as-is. Projections omit 110406 private
locator leaf fields; every other field/type/value, scientific row membership,
label, version and order was independently compared to the source. Original and
output bytes/SHA256 and removed JSON pointers are recorded. No accepted code is
edited and no seed is regenerated. Full JSON byte identity is not claimed for
projections. The exact import and 473 excluded source payloads are enumerated in
`release/IMPORTED_FILES.json`.

Source archives, raw dataset images, pretrained weights, and large
feature/proxy/state/score payloads are external. Path-bearing source metadata is
represented by the documented projections. The Apple
`official_transformations.py` member is excluded pending redistribution license
review. Only the explicitly licensed official CFF schema is newly vendored.

## Checks and results

- PASS: all six original archive identities/CRC/full manifest coverage and direct
  original-to-import byte/projection comparisons.
- PASS: exact CIFAR ordered split, fit/pilot isolation, R1/R2 identity, stage labels
  and versions, mean/paired common evidence; Food six-fold identity, A/B/C
  corruption/request/state locks, qualification isolation and common receipts.
- PASS: eight in-memory scientific lock tampering cases rejected; CFF schema/ORCID
  plus eight invalid fixtures; inventory/hash tampering fixtures; privacy fixtures
  including real UNC paths and standard device namespace distinction.
- PASS: reviewer numerical checks and complete release inventory/privacy checks.
  The public entry point is `python verification/run_all.py`.
- PASS: standalone artifact copy in isolated Python 3.12.14 without site packages;
  see `environment/CLEAN_CHECK.json`. No scientific execution or network required.
- PASS: all 31 externally referenced payload byte/hash identities supplied locally;
  no payload is copied into the artifact. See `environment/EXTERNAL_ASSET_CHECK.json`.
- PASS: numerical table content remained unchanged. Git attributes preserve exact
  artifact bytes across Windows checkouts and the standalone artifact repository.

`release/PUBLIC_INVENTORY.json` and `release/RELEASE_FILES.sha256` seal the final
curated file set. The latter is self-excluded; the inventory excludes itself and
the hash file.

## Artifact scope and licensing

Author-owned code and documentation use MIT (see `LICENSE`). Unresolved
Apple/FastFill source and other third-party payloads remain external under their
original terms. The supplied `sample_current_control.csv` retains its
accepted identity; source and integrity hashes are in the versioned inventory.

The verification entry point is `python verification/run_all.py`. It checks
the frozen evidence and source identities without asserting a new complete
scientific rerun. External datasets, models, original execution gates, and
runtime metadata remain required for full scientific reproduction.
