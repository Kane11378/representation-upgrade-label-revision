#!/usr/bin/env python3
"""Validate CITATION.cff against the vendored official CFF 1.2.0 schema.

The YAML reader accepts this artifact's block mapping/list/scalar profile.
Unsupported YAML syntax fails explicitly. No external dependency is required.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "release" / "citation_schema" / "schema.json"
SCHEMA_GIT_BLOB = "762194bec49e3299bb3acdaec7f24b7e93c602be"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scalar(text):
    text = text.strip()
    require(text and not text.startswith(("&", "*", "!", "[", "{", "|", ">")),
            "unsupported YAML scalar; use a quoted scalar in the block profile")
    if text.startswith('"'):
        value = json.loads(text)
        require(isinstance(value, str), "quoted CFF scalar must be text")
        return value
    if text.startswith("'"):
        require(text.endswith("'") and "'" not in text[1:-1].replace("''", ""),
                "malformed single-quoted YAML scalar")
        return text[1:-1].replace("''", "'")
    require(": " not in text and " #" not in text,
            "ambiguous plain YAML scalar; quote values containing colon or comment syntax")
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.lower() in {"null", "~"}:
        return None
    if re.fullmatch(r"[-+]?(0|[1-9][0-9]*)", text):
        return int(text)
    if re.fullmatch(r"[-+]?[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?", text):
        return float(text)
    return text


def load_block_yaml(text):
    tokens = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        require("\t" not in line, f"tabs are unsupported in CFF indentation at line {number}")
        indent = len(line) - len(line.lstrip(" "))
        require(indent % 2 == 0 and line.strip() not in {"---", "..."},
                f"unsupported CFF YAML layout at line {number}")
        tokens.append([indent, line.strip(), number])

    def parse(index, indent):
        require(index < len(tokens) and tokens[index][0] == indent, "invalid CFF indentation")
        sequence = tokens[index][1].startswith("- ") or tokens[index][1] == "-"
        result = [] if sequence else {}
        while index < len(tokens) and tokens[index][0] == indent:
            _, content, number = tokens[index]
            if sequence:
                require(content.startswith("- ") or content == "-", f"mixed YAML block at line {number}")
                rest = content[1:].strip()
                if not rest:
                    index += 1
                    require(index < len(tokens) and tokens[index][0] == indent + 2,
                            f"missing sequence content at line {number}")
                    value, index = parse(index, indent + 2)
                elif re.match(r"[A-Za-z][A-Za-z0-9-]*:\s", rest) or rest.endswith(":"):
                    tokens[index] = [indent + 2, rest, number]
                    value, index = parse(index, indent + 2)
                else:
                    value = scalar(rest)
                    index += 1
                result.append(value)
            else:
                match = re.fullmatch(r"([A-Za-z][A-Za-z0-9-]*):(?:\s+(.*))?", content)
                require(match is not None, f"unsupported YAML mapping at line {number}")
                key, rest = match.groups()
                require(key not in result, f"duplicate CFF key: {key}")
                index += 1
                if rest is None or not rest.strip():
                    require(index < len(tokens) and tokens[index][0] == indent + 2,
                            f"missing mapping value at line {number}")
                    value, index = parse(index, indent + 2)
                else:
                    value = scalar(rest)
                result[key] = value
            require(index == len(tokens) or tokens[index][0] <= indent,
                    f"unexpected deeper YAML indentation after line {number}")
        return result, index

    require(tokens and tokens[0][0] == 0, "empty CFF or indented root")
    result, end = parse(0, 0)
    require(end == len(tokens) and isinstance(result, dict), "CFF root must be one mapping")
    return result


def validate(value, schema, document, location="$", depth=0):
    require(depth < 100, "CFF schema recursion limit exceeded")
    if "$ref" in schema:
        ref = schema["$ref"]
        require(ref.startswith("#/"), "external CFF schema reference unsupported")
        target = document
        for part in ref[2:].split("/"):
            target = target[part.replace("~1", "/").replace("~0", "~")]
        return validate(value, target, document, location, depth + 1)
    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            successes = 0
            for candidate in schema[keyword]:
                try:
                    validate(value, candidate, document, location, depth + 1)
                except ValueError:
                    continue
                successes += 1
            require(successes >= 1 if keyword == "anyOf" else successes == 1,
                    f"{location}: failed {keyword}")
    type_name = schema.get("type")
    checks = {"object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
              "string": lambda x: isinstance(x, str), "integer": lambda x: type(x) is int,
              "number": lambda x: type(x) in {int, float} and math.isfinite(x)}
    if type_name:
        require(type_name in checks and checks[type_name](value), f"{location}: expected {type_name}")
    if "enum" in schema:
        require(value in schema["enum"], f"{location}: value outside schema enum")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        require(all(key in value for key in schema.get("required", [])), f"{location}: missing required key")
        if schema.get("additionalProperties") is False:
            require(set(value) <= set(properties), f"{location}: unknown CFF key")
        for key, item in value.items():
            if key in properties:
                validate(item, properties[key], document, f"{location}.{key}", depth + 1)
    if isinstance(value, list):
        require(len(value) >= schema.get("minItems", 0), f"{location}: insufficient items")
        if schema.get("uniqueItems"):
            require(len({json.dumps(x, sort_keys=True) for x in value}) == len(value),
                    f"{location}: duplicate items")
        if "items" in schema:
            for index, item in enumerate(value):
                validate(item, schema["items"], document, f"{location}[{index}]", depth + 1)
    if isinstance(value, str):
        require(len(value) >= schema.get("minLength", 0) and len(value) <= schema.get("maxLength", math.inf),
                f"{location}: string length outside schema bounds")
        if "pattern" in schema:
            require(re.search(schema["pattern"], value) is not None, f"{location}: schema pattern mismatch")
        if schema.get("format") == "date":
            try:
                datetime.date.fromisoformat(value)
            except ValueError:
                raise ValueError(f"{location}: invalid calendar date") from None
        if schema.get("format") == "uri":
            parsed = urlsplit(value)
            require(bool(parsed.scheme) and not re.search(r"\s", value), f"{location}: invalid URI")
            if parsed.scheme in {"https", "http", "ftp", "sftp"}:
                require(bool(parsed.netloc), f"{location}: URI lacks authority")
    if type(value) in {int, float}:
        require(schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf),
                f"{location}: number outside schema bounds")


def verify_orcid(value):
    digits = urlsplit(value).path.strip("/").replace("-", "")
    require(re.fullmatch(r"[0-9]{15}[0-9X]", digits) is not None, "invalid ORCID structure")
    total = 0
    for digit in digits[:-1]:
        total = (total + int(digit)) * 2
    check = (12 - total % 11) % 11
    require(digits[-1] == ("X" if check == 10 else str(check)), "ORCID checksum mismatch")


def main():
    payload = SCHEMA_PATH.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload).hexdigest()
    require(blob == SCHEMA_GIT_BLOB, "vendored CFF schema differs from the official version")
    schema = json.loads(payload)
    metadata = load_block_yaml((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    validate(metadata, schema, schema)
    for author in metadata["authors"]:
        if "orcid" in author:
            verify_orcid(author["orcid"])
    print("PASS: CITATION.cff validates against the exact official CFF 1.2.0 schema and ORCID checksum.")


if __name__ == "__main__":
    main()
