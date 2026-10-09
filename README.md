# Reproducibility Artifact

Code, frozen protocol evidence, and derived result tables for:

> **When More Faithful Label Corrections Look Worse: Diagnosing Classifier Maintenance after Representation Upgrades**

The artifact distinguishes label-revision response from the composed maintenance
action after a representation upgrade. It includes accepted CIFAR-100 and
Food-101 confirmation code, frontend checks, interaction analysis, protocol
locks, and machine-readable tables supporting the paper and Supplement.

The versioned [v1.0.1 research artifact](https://github.com/Kane11378/representation-upgrade-label-revision/tree/v1.0.1)
provides a fixed reference for the code and supporting evidence.
[CITATION.cff](CITATION.cff) gives the software and manuscript citation details.

## Quickstart

From the repository root, run with Python 3.10 or later:

```sh
python verification/run_all.py
```

The checks use the Python standard library and require no GPU, datasets,
pretrained weights, external arrays, or network access. The final output is:

```text
ALL_LIGHTWEIGHT_CHECKS_PASS
```

See [REVIEWER_QUICKSTART.md](REVIEWER_QUICKSTART.md) for the checks and
[PAPER_TO_ARTIFACT_MAP.csv](PAPER_TO_ARTIFACT_MAP.csv) for the claim-by-claim
crosswalk. [STATUS.md](STATUS.md) describes the verification scope.

## Results and protocols

| Evidence | Files |
|---|---|
| Primary CIFAR and Food-101 results | [CIFAR table](tables/cifar_final_primary.csv), [Food-101 table](tables/food101_external_primary.csv), [Food-101 streams](tables/food101_external_streams.csv) |
| Frontend stress tests | [Stress table](tables/frontend_stress.csv), [frontend code](code/README.md) |
| Interaction decomposition | [Interaction table](tables/interaction_decomposition.csv), [saved-score analysis](code/analyses/analyze_saved_scores.py) |
| Endpoint comparisons | [Endpoint table](tables/endpoint_summary.csv), [retained control](tables/sample_current_control.csv) (Supplement Table S16) |
| Protocol evidence | [Protocol guide](protocol/README.md), [CIFAR summary](protocol/cifar_final/STUDY_LOCK.json), [Food-101 summary](protocol/food101_external/STUDY_LOCK.json) |
| Accepted delivery identities | [Package hashes](tables/frozen_package_hashes.csv), [source provenance](PROVENANCE.md) |

The protocol summaries support checks of counts, request arithmetic, encoder
selection, and shared backend hyperparameters. Row-level evidence is under each
protocol's `accepted/` directory. The verifier checks membership, labels, request
order, and common evidence without regenerating seeds.

For the interaction analysis, the released table satisfies:

```text
delta_composed
= delta_refresh
+ delta_revision_response
+ delta_cross_term
```

The table check does not require saved scores. Running the analysis itself
requires the original frozen score package. The script writes an English-language `REPORT.md`; [PROVENANCE.md](PROVENANCE.md)
records its source identity, released hash and numerical scope.

## Reproducibility scope

Accepted confirmation code and numerical CSVs retain their original bytes.
Path-bearing JSON locks are documented privacy projections: scientific values,
row identities, and ordering are preserved, while locator fields are omitted.
[IMPORTED_FILES.json](release/IMPORTED_FILES.json) records original and projected
hashes and the omitted field pointers. These projections support verification
and do not replace the original runners' path-bound execution metadata.

The release checks frozen evidence and code identities. A fully portable
one-command end-to-end scientific rerun remains unvalidated. It requires external
assets, the accepted runtime and source gates, and a compatible experiment
layout. [Environment instructions](environment/README.md) give the recorded
hardware, libraries, phase commands, and remaining prerequisites.

Datasets, pretrained weights, source archives, and large cached arrays are
external. [Third-party asset instructions](third_party/ASSET_INSTRUCTIONS.md)
provide acquisition guidance and identity records. The unresolved FastFill
architecture source is excluded pending redistribution-license review.

## Availability, integrity, and licensing

[DATA_AND_CODE_AVAILABILITY.md](DATA_AND_CODE_AVAILABILITY.md) gives the versioned
availability statement. [SOURCE_VERIFICATION.md](SOURCE_VERIFICATION.md) records the
source archive and artifact integrity checks. [PUBLIC_INVENTORY.json](release/PUBLIC_INVENTORY.json)
and [RELEASE_FILES.sha256](release/RELEASE_FILES.sha256) record the complete
artifact membership, byte counts, and hashes.

Author-owned code and documentation use the [MIT License](LICENSE); see
[LICENSE_DECISION.md](LICENSE_DECISION.md) for its scope. Third-party material
retains its upstream terms, including the CFF schema's CC-BY-4.0 attribution.

The artifact supports the paper's bounded empirical claims. It does not establish
universal maintenance superiority, universal validity for nonlinear models, or
certified unlearning.
