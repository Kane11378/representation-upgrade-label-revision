# Artifact Provenance

The artifact contains accepted source code, transcriptions of frozen numerical
evidence, and utilities that verify their identities and internal consistency.
Verification utilities perform no scientific fitting or model inference.

Original experiment sources and delivery archives are not redistributed. Their
identities are recorded below and in the source ledgers; a recorded source
identity does not imply that the source archive is publicly available.

## Frontend and analysis sources

The SHA256 values below identify the accepted original source bytes.
The four frontend files in this artifact match those original identities.

The additional `code/frontends/frontend_interface.py` compatibility adapter
reexports `fastfill_interface.py` under the module name used by the accepted
check. It does not replace or modify either original file. Its own release hash
is in [RELEASE_FILES.sha256](release/RELEASE_FILES.sha256).

| Artifact path | Accepted source component | Original SHA256 | Current status |
|---|---|---|---|
| `code/frontends/procrustes_alignment.py` | Procrustes alignment implementation | `59be9742d62f67aa6b1c3c47e139932fa4686f1916a8cfc8b8c2c52c353bb78f` | Exact source bytes |
| `code/frontends/check_procrustes.py` | Procrustes alignment checks | `cbc6fe602f5332009dfef3db9c06c70d5d4866a403cfd52c813df9d8478c50e2` | Exact source bytes |
| `code/frontends/fastfill_interface.py` | FastFill integration interface | `d75e64ea018833458107bc33d8ca60897dc1d52b50441e15ed476156aa8d34d7` | Exact source bytes |
| `code/frontends/check_fastfill_interface.py` | FastFill parity and integration checks | `7d6eb5340fa1fb18d5485df3f9aa778fccd4cb2cb78ad439eb51d4c8521f740e` | Exact source bytes |
| `code/analyses/analyze_saved_scores.py` | Saved-score interaction decomposition | `41321f9c7a8e570e07df3d96ab8a60ed47b017060506be08fbe3850c7af9e8a1` | English report adaptation |

The saved-score analysis produces an English-language `REPORT.md`. Its
numerical operations, input handling and CSV outputs follow the accepted
implementation; the released script is not byte-identical because its report
text and output filename differ. The original source SHA256 is listed above;
the released script's SHA256 is in [RELEASE_FILES.sha256](release/RELEASE_FILES.sha256).
Checks on the frozen score inputs found byte-identical outputs for the three
CSV results and `STATUS.json`. These checks do not execute encoders or
reproduce the full confirmation pipeline.

## Accepted archive imports

[SOURCE_ARCHIVES.json](release/SOURCE_ARCHIVES.json) records the six accepted ZIP
identities, filename checks, byte counts, SHA256 values, CRC checks, and complete
internal-manifest verification performed before import. The archive identifiers
are stable provenance keys:

| Archive ID | Accepted evidence |
|---|---|
| `legacy_alignment` | Alignment bridge source and supporting environment record |
| `cifar_protocol` | Final held-out CIFAR protocol |
| `cifar_confirmation` | Final held-out CIFAR confirmation |
| `food_protocol` | Six-fold Food-101 protocol |
| `food_qualification` | Food-101 encoder qualification |
| `food_confirmation` | Food-101 external confirmation |

The alignment download used a basename without the requested `(1)` suffix.
Its complete byte identity matches the accepted package; the alias is recorded
in the source ledger.

[IMPORTED_FILES.json](release/IMPORTED_FILES.json) maps each imported member to its
archive, original member name, source hash, and output hash. It distinguishes
`exact_bytes` from `privacy_projection`. Accepted execution code under
`code/accepted/`, including both `current_core.py` copies, is unedited and
byte-identical to the ZIP members. Both core copies retain SHA256
`0b28ae97c4fcd65eb901938e06a8642a899e5c32208b239da7b4ac64101d6bd7`.
Original source manifests anchor the member identities.

The archive and import ledgers record source archive/member identities,
numerical fields, selected-file hashes and verification receipts. They also
include SHA256 identities of the original source ledgers. Public metadata views
omit machine-local locator fields while retaining the scientific values and
original source/member names. The imported scientific files and recorded
identity relationships are unchanged.

For path-bearing JSON, the import utility omits only locator-bearing string
leaves, records each removed JSON pointer, and compares every retained nonpath
value. Scientific scalars, labels, row membership, request order, and seeds are
preserved. Exact imported CSVs require no transformation. Full original JSON
byte identity is not claimed for privacy projections.

## Numerical evidence

The reviewer tables are transcriptions of frozen accepted values, rather than
newly fitted results. Their source evidence is identified by study and analysis:

| Table | Accepted evidence |
|---|---|
| `tables/cifar_final_primary.csv` | Final held-out CIFAR confirmation |
| `tables/food101_external_primary.csv` | Food-101 external confirmation |
| `tables/food101_external_streams.csv` | Food-101 external confirmation by stream and event |
| `tables/frontend_stress.csv` | Accepted affine, Procrustes, and FastFill frontend studies summarized in the paper and Supplement |
| `tables/interaction_decomposition.csv` | Frozen saved-score interaction decomposition |
| `tables/endpoint_summary.csv` | Final CIFAR and Food-101 endpoint comparisons |
| `tables/frozen_package_hashes.csv` | Accepted delivery identities reported in Supplement S7 |
| `tables/sample_current_control.csv` | Accepted retained current-sample control, copied byte-for-byte and mapped to Supplement Table S16 |

[PAPER_TO_ARTIFACT_MAP.csv](PAPER_TO_ARTIFACT_MAP.csv) supplies the scientific
crosswalk. Archive and member identities for imported tables are in the import
ledger. Stable experiment and member names in those records identify the actual
sources and are retained.

## Protocol summaries and verification

`protocol/cifar_final/STUDY_LOCK.json` summarizes the accepted CIFAR protocol
freeze and final confirmation. `protocol/food101_external/STUDY_LOCK.json`
summarizes the accepted Food-101 protocol freeze, encoder qualification, and
external confirmation. These summaries are distinct from the imported row-level
locks in the `accepted/` subdirectories.

Utilities under `verification/` check frozen arithmetic, protocol identities,
source provenance, citation metadata, and artifact integrity. They do not train
models, access hidden labels or features, choose hyperparameters, or create
scientific evidence. [SOURCE_VERIFICATION.md](SOURCE_VERIFICATION.md) records the historical
source-import checks and their limits.

Historical runtime evidence is in `environment/RELEASE_ENVIRONMENT.json`; it is
not a portable scientific dependency lock. A full scientific rerun still
requires external assets and compatible execution metadata and source gates.
