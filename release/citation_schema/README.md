# Citation File Format schema

`schema.json` is an unchanged copy of the Citation File Format project
contributors' CFF 1.2.0 JSON Schema, retrieved from the immutable version tag:

- Source: https://github.com/citation-file-format/citation-file-format/blob/1.2.0/schema.json
- Git blob identity: `762194bec49e3299bb3acdaec7f24b7e93c602be`
- License: Creative Commons Attribution 4.0 International; upstream license text
  and disclaimer are retained in `LICENSE.txt`.
- License source: https://github.com/citation-file-format/citation-file-format/blob/1.2.0/LICENSE

This schema's existing third-party license applies only to these upstream
materials. It does not select a license for the author-owned artifact.

`verification/verify_citation.py` implements every validation keyword used by
this schema using the Python standard library. Its YAML reader deliberately
supports the artifact's block mapping/list/scalar profile and rejects unsupported
YAML constructs instead of silently accepting them. The reader preserves quoted
date values as strings, as the official schema requires.
