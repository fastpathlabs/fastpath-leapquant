"""Shared helpers: canonical JSON, content hashes, strict JSON/JSONL reading and writing, finite-number checks.

Everything here is standard library only, so the validator and the independent checker run on any Python 3.10+.
"""
from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


class StrictJsonError(ValueError):
    """Malformed JSON, or JSON that uses NaN / Infinity (never valid evidence)."""


def _reject_constant(name: str) -> Any:
    raise StrictJsonError(f"non-finite JSON constant {name!r}")


def loads_strict(text: str) -> Any:
    """json.loads that refuses NaN, Infinity and -Infinity instead of turning them into floats."""
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise StrictJsonError(str(exc)) from exc


def dumps(obj: Any, *, indent: int | None = 1) -> str:
    """JSON text with NaN/Infinity forbidden: a missing value must be null with a reason, never NaN."""
    return json.dumps(obj, indent=indent, ensure_ascii=False, allow_nan=False)


def canonical_json(obj: Any) -> str:
    """Key-sorted, whitespace-free JSON: the input of every content hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_hash(obj: Any) -> str:
    """Hash of a configuration object (canonical JSON). Two arms with the same hash are the same configuration."""
    return sha256_text(canonical_json(obj))


def open_text(path: str | os.PathLike, mode: str = "rt"):
    """Opens .gz files transparently."""
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8") if "t" in mode else gzip.open(path, mode)
    return open(path, mode, encoding="utf-8")


def iter_jsonl(path: str | os.PathLike) -> Iterator[tuple[int, Any]]:
    """Yields (line number, record). Raises StrictJsonError naming the line on malformed or non-finite JSON."""
    with open_text(path) as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield number, loads_strict(line)
            except StrictJsonError as exc:
                raise StrictJsonError(f"{path}:{number}: {exc}") from exc


def read_jsonl(path: str | os.PathLike) -> list[Any]:
    return [record for _, record in iter_jsonl(path)]


def write_jsonl(path: str | os.PathLike, records: Iterable[Any]) -> int:
    count = 0
    with open_text(path, "wt") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def read_json(path: str | os.PathLike) -> Any:
    with open_text(path) as handle:
        return loads_strict(handle.read())


def write_json(path: str | os.PathLike, obj: Any) -> None:
    """Writes atomically (temporary file + rename), so a crash never leaves half a manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(dumps(obj) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_utc(text: str) -> _dt.datetime:
    text = text.strip().replace("Z", "+00:00")
    value = _dt.datetime.fromisoformat(text)
    if value.tzinfo is None:
        value = value.replace(tzinfo=_dt.timezone.utc)
    return value


def null(reason: str, **extra: Any) -> dict:
    """A missing value with its reason. Never 0, never NaN, never omitted silently."""
    out = {"value": None, "null_reason": reason}
    out.update(extra)
    return out


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def harness_root() -> Path:
    return Path(__file__).resolve().parents[1]


def relpath(path: str | os.PathLike, start: str | os.PathLike | None = None) -> str:
    start = Path(start) if start else repo_root()
    try:
        return str(Path(path).resolve().relative_to(start.resolve()))
    except ValueError:
        return str(path)
