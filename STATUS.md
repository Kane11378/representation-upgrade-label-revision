# Reproducibility Status

The public [v1.0.1 artifact](https://github.com/Kane11378/representation-upgrade-label-revision/tree/v1.0.1)
provides the code, derived data and accepted evidence supporting the manuscript.

## Evidence and lightweight checks

The artifact provides:

- paper arithmetic, evidence coverage, correction-size, interaction, endpoint,
  and conditional bootstrap checks;
- frozen row membership, request order, corruption, and shared-evidence identities;
- accepted CIFAR and Food-101 execution code and source-member hashes;
- inventory and SHA256 verification, privacy checks, and offline CFF 1.2.0 validation;
- historical runtime records and isolated lightweight-verification evidence.

Both accepted `current_core.py` files retain SHA256
`0b28ae97c4fcd65eb901938e06a8642a899e5c32208b239da7b4ac64101d6bd7`.
Numerical CSVs and accepted execution code retain their original bytes. The
saved-score analysis produces `REPORT.md`; numerical operations and source
identities are recorded in [PROVENANCE.md](PROVENANCE.md).

Protocol summaries remain distinct from row-level locks. Path-bearing JSON locks
are documented privacy projections, with original member hashes, removed locator
field pointers, retained-value comparisons, and output hashes. Scientific row
values and ordering are preserved without regenerating seeds. These projections
support identity verification and do not directly satisfy the original runners'
path-bound execution gates.

## Full scientific rerun

Datasets, pretrained weights, feature/proxy/state/score arrays, and the unresolved
FastFill architecture source are external. Accepted runtime evidence distinguishes
CIFAR CUDA encoding from CPU confirmation and Food-101 CPU execution. A portable
scientific dependency lock and compatible execution metadata still require
validation. See [environment/README.md](environment/README.md),
[EXTERNAL_ASSETS.json](third_party/EXTERNAL_ASSETS.json), and
[SOURCE_VERIFICATION.md](SOURCE_VERIFICATION.md).

The release verifies frozen evidence and source identities. It does not train new
models, tune methods, select alternate requests, or certify a fully portable
end-to-end scientific rerun. The accepted `sample_current` table retains its
original bytes and maps to Supplement Table S16.

Author-owned code and documentation use MIT; third-party material retains its
upstream terms.
