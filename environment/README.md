# Environment and clean-check record

## Lightweight release verification

Python 3.10+ and the standard library are sufficient for:

```sh
python verification/run_all.py
```

`CLEAN_CHECK.json` records historical lightweight verification in an isolated Python 3.12.14 environment created without pip or site packages. That check did not execute scientific runners. `RELEASE_ENVIRONMENT.json` separates historical accepted runtime evidence from the current release-tool distribution inventory. Observing an installed distribution does not prove an accepted imported-version identity or full rerun compatibility.

## Historical accepted runtime evidence

- CIFAR DINO/FastFill encoding: Python 3.12.14, NumPy 2.5.2, PyTorch 2.11.0+cu128, CUDA 12.8, Pillow 12.3.0, NVIDIA GeForce RTX 5090. Eight threads; seed 641001; FP32/eval; TF32 disabled; highest matmul precision. The original run records cuDNN benchmark=false, cuDNN deterministic=false and deterministic algorithms=false. Preserve these actual settings rather than claiming fully deterministic GPU execution.
- CPU backend environment metadata: NumPy 2.5.3, SciPy 1.18.1, threadpoolctl 3.7.0, PyTorch distribution 2.14.1, TorchVision distribution 0.29.1. Frozen pre-execution filesystem metadata records these installed versions; it is not an imported-version gate for every confirmation phase. The release-tool environment observes the same versions separately.
- Food encoder runtime gate: Python 3.12.14, PyTorch 2.14.1+cpu and TorchVision 0.29.1+cpu with their accepted git identities. CPU FP32/eval/no_grad; eight intra-op and BLAS threads, one inter-op thread; batch size 32, workers=0, shuffle=false, highest matmul precision. Accepted encoder code enables deterministic algorithms and uses seed 0 before replacing all pretrained parameters. NumPy/SciPy backend imports use FP64 and fixed eight-thread BLAS limits where specified by the accepted code.

Original bytes/hashes and settings are sourced in `RELEASE_ENVIRONMENT.json`, `protocol/cifar_final/accepted/FEATURE_CACHE_LOCK.json` and `protocol/food101_external/accepted/confirmation/RUNTIME_GATE.json`. The latter JSON files are documented privacy projections. `release/source_environment/legacy_alignment_ENVIRONMENT.json` is historical supporting provenance, not the final full-rerun dependency lock.

## Full-rerun prerequisites and phase commands

The unchanged final code is under `code/accepted/cifar_final/` and `code/accepted/food101_external/`. It still computes its experiment root from its file location and checks original source/package/runtime identity gates. The artifact layout and privacy-projected JSON support release verification, but do not directly replace the original experiment layout. No relocation adapter or reconstructed runner is represented as accepted code.

A full rerun requires the original verified packages, a fresh writable experiment layout, original execution/source/runtime metadata, exact dependencies and third-party assets. The accepted scripts make one-shot output files and refuse existing outputs; archive the prior accepted evidence separately before using a fresh output layout. A portable locator-binding layer that preserves the frozen scientific identities remains a prerequisite for a future fully portable scientific rerun. Do not weaken the original gates or edit the accepted runner merely to make it run.

Once those prerequisites and identity gates are satisfied in the restored original experiment layout, the accepted confirmation phases are:

```sh
# From fastfill_final_unseen_confirmation_01:
python code/confirmation.py build
python code/confirmation.py evaluate
# From food101_cross_domain_revision_confirmation_01:
python code/confirmation_bridge_v2.py build
python code/confirmation_bridge_v2.py evaluate
```

These phase commands are conditional on the original prerequisites; they have not been certified as portable artifact commands. The released code preserves alpha=.25, ridge=1, shrink=.1 and all frozen evidence/request choices. The trained FastFill checkpoint is reused without new training. To reproduce encoder selection, retain the original Food qualification gate/runner and folds 0/1 only; confirmation uses folds 2/3/4 with fold 5 as common evaluator.

External byte identities are in `third_party/EXTERNAL_ASSETS.json`; detailed frozen cache/score/state identities remain in the projected locks. CIFAR confirmation needs global IDs, B features and the frozen FastFill history proxy, plus original locked ancillary metadata. Food needs eight exact ResNet18/Swin-T fold 2-5 caches (IDs, labels and features), the frozen protocol, selected encoder and source/runtime gates. Saved scores are needed for offline bootstrap; saved state/proxy/feature arrays for the other frozen analyses. Re-encoding additionally requires original datasets, weights, preprocessing/sentinel identity and GPU/CPU runtime gates. Third-party architecture code licensing is unresolved.

A local JSON map of external asset IDs to files can be checked without loading models:

```sh
python verification/verify_external_assets.py asset-map.json
# Require all identities listed in the external inventory:
python verification/verify_external_assets.py asset-map.json --require-all
```

Keep that local map outside the released tree. Full original JSON byte identity, a portable scientific dependency lock, clean-machine full scientific execution and numerical output reproduction remain unvalidated. These are limitations of full scientific reproduction; this release provides the accepted code, metadata, derived tables and lightweight checks without claiming full scientific execution.
