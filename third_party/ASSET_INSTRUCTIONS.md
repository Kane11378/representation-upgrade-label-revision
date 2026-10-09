# Third-Party Asset Instructions

This reproducibility artifact does not redistribute third-party datasets or pretrained model weights.

Obtain the following from their official distribution channels:

- CIFAR-100 and CIFAR-100N;
- Food-101;
- DINOv2-S/14 and DINOv2-B/14 pretrained encoders;
- TorchVision ResNet18, ConvNeXt-Tiny, Swin-T and ViT-B/16 ImageNet-pretrained weights used in qualification.

The imported frozen metadata and `EXTERNAL_ASSETS.json` record, where the accepted packages provide them:
- exact library/model identifiers;
- preprocessing transforms;
- feature dimensions;
- dataset split/manifests or stable hashes;
- any accepted sentinel/hash checks.

This file is intentionally license-conservative: dataset images and pretrained weights should not be copied into the public repository unless their licenses explicitly permit redistribution.

## External payload identities

`EXTERNAL_ASSETS.json` lists frozen bytes/SHA256 and original ZIP/member/pointer
provenance for dataset archives, pretrained weights, CIFAR features/proxy,
Food fold 2-5 feature caches and saved score/prediction outputs. Further state
and ancillary identities remain in `protocol/*/accepted/`. The inventory is an
identity record; its output entries are not all prerequisite inputs to every
phase. All payloads remain external, including the trained epoch-79 FastFill
checkpoint. The inventory does not download assets or train models.

Use official URLs/enums/transforms preserved in the selected-encoder and model
candidate locks. Food image-relative IDs and image hashes are preserved in the
fold/qualification/deployment records. Do not select a different pretrained
payload merely because it shares an architecture name.

A local asset-ID to filename JSON map can be checked with
`python verification/verify_external_assets.py asset-map.json`; use
`--require-all` to require the complete listed identity inventory. Keep that
local map outside the artifact tree. Byte count and complete SHA256 are checked without
deserializing weights or computing performance.

The original CIFAR third-party architecture member
`code/official_transformations.py` has an Apple copyright header and no
redistribution license in the frozen ZIP. It is intentionally excluded pending
license review; its source identity is retained in the import exclusion ledger.
The existing FastFill interface is author-owned integration code and remains part of the artifact. Unresolved upstream source is not redistributed in this release.

The official CFF 1.2.0 schema is the only newly vendored third-party component;
its upstream CC-BY-4.0 license and attribution are in
`release/citation_schema/`. This does not license the author's code.
