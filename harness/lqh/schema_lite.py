"""A small JSON Schema validator (the draft 2020-12 keywords the harness schemas use), standard library only.

Supported: type, enum, const, required, properties, additionalProperties, patternProperties, items, prefixItems,
minItems, maxItems, uniqueItems, minimum, maximum, exclusiveMinimum, exclusiveMaximum, minLength, maxLength,
pattern, anyOf, oneOf, allOf, not, $ref (local "#/$defs/..." and "#/definitions/..."), if/then/else.
Numbers must be finite. Unknown keywords are ignored, as JSON Schema requires; `lqh` schemas use only the above.
When the `jsonschema` package is installed the tests also run it against the same schemas.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .util import harness_root, loads_strict

_TYPE_CHECKS = {
    "null": lambda v: v is None,
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


class SchemaError(ValueError):
    pass


def _resolve(root: dict, ref: str) -> dict:
    if not ref.startswith("#/"):
        raise SchemaError(f"only local references are supported: {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def validate(instance: Any, schema: dict, root: dict | None = None, path: str = "$") -> list[str]:
    """Returns a list of error messages (empty when valid)."""
    root = root if root is not None else schema
    errors: list[str] = []
    if schema is True or schema == {}:
        return errors
    if schema is False:
        return [f"{path}: no value is allowed here"]
    if "$ref" in schema:
        errors += validate(instance, _resolve(root, schema["$ref"]), root, path)
    if isinstance(instance, float) and not math.isfinite(instance):
        errors.append(f"{path}: non-finite number")
        return errors
    expected = schema.get("type")
    if expected is not None:
        names = expected if isinstance(expected, list) else [expected]
        if not any(_TYPE_CHECKS[name](instance) for name in names):
            errors.append(f"{path}: expected {'/'.join(names)}, got {type(instance).__name__}")
            return errors
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in {schema['enum']}")
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: must equal {schema['const']!r}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {instance} <= exclusiveMinimum {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: {instance} >= exclusiveMaximum {schema['exclusiveMaximum']}")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{path}: {instance!r} does not match {schema['pattern']!r}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            seen = [json.dumps(item, sort_keys=True) for item in instance]
            if len(seen) != len(set(seen)):
                errors.append(f"{path}: items are not unique")
        prefix = schema.get("prefixItems", [])
        for index, item in enumerate(instance):
            if index < len(prefix):
                errors += validate(item, prefix[index], root, f"{path}[{index}]")
            elif "items" in schema:
                errors += validate(item, schema["items"], root, f"{path}[{index}]")
    if isinstance(instance, dict):
        for name in schema.get("required", []):
            if name not in instance:
                errors.append(f"{path}: missing required field {name!r}")
        properties = schema.get("properties", {})
        patterns = schema.get("patternProperties", {})
        for key, value in instance.items():
            matched = False
            if key in properties:
                matched = True
                errors += validate(value, properties[key], root, f"{path}.{key}")
            for pattern, subschema in patterns.items():
                if re.search(pattern, key):
                    matched = True
                    errors += validate(value, subschema, root, f"{path}.{key}")
            if not matched and "additionalProperties" in schema:
                extra = schema["additionalProperties"]
                if extra is False:
                    errors.append(f"{path}: unexpected field {key!r}")
                elif isinstance(extra, dict):
                    errors += validate(value, extra, root, f"{path}.{key}")
    for sub in schema.get("allOf", []):
        errors += validate(instance, sub, root, path)
    if "anyOf" in schema:
        if not any(not validate(instance, sub, root, path) for sub in schema["anyOf"]):
            errors.append(f"{path}: matches none of anyOf")
    if "oneOf" in schema:
        matches = sum(1 for sub in schema["oneOf"] if not validate(instance, sub, root, path))
        if matches != 1:
            errors.append(f"{path}: matches {matches} of oneOf (exactly one required)")
    if "not" in schema and not validate(instance, schema["not"], root, path):
        errors.append(f"{path}: matches a forbidden schema")
    if "if" in schema:
        if not validate(instance, schema["if"], root, path):
            if "then" in schema:
                errors += validate(instance, schema["then"], root, path)
        elif "else" in schema:
            errors += validate(instance, schema["else"], root, path)
    return errors


_CACHE: dict[str, dict] = {}


def load_schema(name: str) -> dict:
    """Loads harness/schemas/<name>.schema.json."""
    if name not in _CACHE:
        path = harness_root() / "schemas" / f"{name}.schema.json"
        _CACHE[name] = loads_strict(Path(path).read_text(encoding="utf-8"))
    return _CACHE[name]


def check(instance: Any, name: str) -> list[str]:
    return validate(instance, load_schema(name))
