# Reviewer Quickstart

To check the [v1.0.1 release](https://github.com/Kane11378/representation-upgrade-label-revision/tree/v1.0.1), run:

```sh
git clone --branch v1.0.1 https://github.com/Kane11378/representation-upgrade-label-revision.git
cd representation-upgrade-label-revision
python verification/run_all.py
```

The verifier uses only the Python standard library (Python 3.10+).
No GPU, datasets, pretrained models, external arrays or network are needed.
Expected final output: `ALL_LIGHTWEIGHT_CHECKS_PASS`.

Checks cover paper arithmetic, Food 6/6 response direction, interaction identity,
endpoint direction, protocol-summary arithmetic, exact-evidence coverage,
correction-size sensitivity and conditional interval signs; exact CIFAR/Food
membership and request/evidence identities, encoder-qualification isolation,
accepted execution-code hashes, source manifests and import provenance; CFF 1.2.0
schema/ORCID constraints, public privacy hygiene and complete inventory/SHA256
identities. The accepted `sample_current_control.csv` is included and mapped in
`PAPER_TO_ARTIFACT_MAP.csv`; it is copied without new calculation.

Root Git administration metadata is outside the artifact inventory. Unexpected
artifact files, symlinks, and transient caches are rejected.

To prepare a sealed copy outside the source tree, use an empty destination:

```sh
python -B release/build_candidate.py /path/to/empty-artifact-directory
python -B /path/to/empty-artifact-directory/verification/run_all.py
```

In a Git checkout the builder uses reviewed tracked paths; untracked files and
Git administration are excluded. A standalone artifact must first pass its
existing inventory and SHA256 checks. No datasets, weights or source archives
are added by the builder. The builder writes new seals only in the destination,
then runs the lightweight checks there. Source files and source seals remain
unchanged. The resulting copy has the same numerical evidence and accepted
scientific code.

`release/SOURCE_ARCHIVES.json` records historical accepted package verification.
`release/IMPORTED_FILES.json` distinguishes byte-exact imports from privacy
projections. Both are reader-oriented metadata views with original historical
ledger hashes; archive/member identities and imported-file receipts are retained.
Holders of the original packages can use the import utility to repeat
CRC/SHA/internal-manifest and retained-value comparisons. Original path-bearing
metadata, source ZIPs and third-party/cache payloads are excluded.

Author-owned code/documentation is MIT licensed. The official CFF schema retains
CC-BY-4.0 attribution. Unresolved Apple/FastFill architecture source is excluded.
Checking frozen evidence and code identities does not constitute a new encoder,
model or full scientific rerun. Full prerequisites and limitations are documented
in `environment/README.md` and `third_party/ASSET_INSTRUCTIONS.md`.
