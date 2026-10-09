# Scientific Code

## Frontends

- `frontends/procrustes_alignment.py`
- `frontends/check_procrustes.py`
- `frontends/fastfill_interface.py`
- `frontends/check_fastfill_interface.py`

These files preserve the accepted original source bytes for alignment and
FastFill integration checks. [PROVENANCE.md](../PROVENANCE.md) records their
immutable source hashes.

`frontends/frontend_interface.py` supplies the historical module name expected
by the FastFill check and reexports the published interface. It adds no numerical
operations; the four accepted frontend files remain unchanged. With NumPy and
PyTorch installed, run from the repository root:

```sh
python -B code/frontends/check_fastfill_interface.py --help
python -B code/frontends/check_fastfill_interface.py --upstream-dir /path/to/ml-fct --out /path/to/RESULTS.json
```

The parity check requires the two upstream files at their pinned Git blob
identities. It performs synthetic component checks without downloading data or
training models; it is not a full FastFill reproduction.

## Saved-score analysis

`analyses/analyze_saved_scores.py` performs read-only interaction decomposition
on the frozen score inputs and produces `REPORT.md`. The numerical operations
and CSV outputs follow the accepted implementation. The report presentation
is distinct from the original source; original and released code hashes are
recorded in [PROVENANCE.md](../PROVENANCE.md) and the release ledger.

## Frozen confirmation and qualification code

- `accepted/cifar_final/`: shared core, final confirmation, and encoding/proxy helpers;
- `accepted/food101_external/`: shared core, final confirmation, Procrustes, and protocol gate;
- `accepted/food101_qualification/`: encoder qualification runner and protocol gate.

Every file in these directories is an unedited original ZIP member. Original and
output byte counts and SHA256 values are recorded in
[IMPORTED_FILES.json](../release/IMPORTED_FILES.json). Both `current_core.py`
copies retain SHA256
`0b28ae97c4fcd65eb901938e06a8642a899e5c32208b239da7b4ac64101d6bd7`.

Datasets, checkpoints, and pretrained model weights are external.
`official_transformations.py` is excluded because the frozen ZIP does not
establish redistribution permission; its source identity is retained.

The accepted scripts enforce their original experiment-layout and source gates.
Direct execution from this artifact layout is not a certified full scientific
rerun. [Environment instructions](../environment/README.md) describe the
prerequisites and conditional phase commands. Frozen packages remain the source
of truth for accepted execution code.
