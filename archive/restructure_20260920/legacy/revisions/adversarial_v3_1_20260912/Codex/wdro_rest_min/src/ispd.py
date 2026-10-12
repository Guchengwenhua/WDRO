"""Design-aware ISPD parsing without evaluating or exposing model metrics."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .geometry import InvalidNetError, normalize_net


DESIGN_PATTERN = re.compile(r"^ispd(18|19)_test(\d+)\.input$")
HEADER_PATTERN = re.compile(r"^net\s+(\S+)\s+(\S+)\s+(\d+)\s*$")
DEGREE_FILE_PATTERN = re.compile(r"^degree_(\d+)\.txt$")


@dataclass(frozen=True)
class SourceNet:
    net_name: str
    declared_degree: int
    pins: np.ndarray
    source_line: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def design_sort_key(path: Path) -> tuple[int, int]:
    match = DESIGN_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"not a base ISPD design directory: {path}")
    return int(match.group(1)), int(match.group(2))


def discover_base_designs(root: str | Path) -> list[Path]:
    benchmark_root = Path(root)
    designs = [path for path in benchmark_root.iterdir() if path.is_dir() and DESIGN_PATTERN.match(path.name)]
    return sorted(designs, key=design_sort_key)


def discover_excluded_variants(root: str | Path) -> list[Path]:
    benchmark_root = Path(root)
    return sorted(
        [path for path in benchmark_root.iterdir() if path.is_dir() and path.name.endswith("_metal5.input")],
        key=lambda path: path.name,
    )


def parse_degree_file(path: str | Path) -> Iterator[SourceNet]:
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        header = HEADER_PATTERN.match(stripped)
        if header is None:
            index += 1
            continue
        source_line = index + 1
        net_name = header.group(1)
        declared_degree = int(header.group(3))
        index += 1
        coordinates: list[list[float]] = []
        while index < len(lines) and lines[index].strip() != "END":
            parts = lines[index].split()
            if len(parts) != 2:
                raise InvalidNetError("malformed_source_block", f"{source}:{index + 1}")
            try:
                coordinates.append([float(parts[0]), float(parts[1])])
            except ValueError as error:
                raise InvalidNetError("malformed_source_block", f"{source}:{index + 1}") from error
            index += 1
        if index >= len(lines):
            raise InvalidNetError("malformed_source_block", f"missing END after {source}:{source_line}")
        index += 1
        pins = np.asarray(coordinates, dtype=np.float64)
        if pins.shape != (declared_degree, 2):
            raise InvalidNetError(
                "source_degree_mismatch",
                f"{source}:{source_line} declared {declared_degree}, parsed {len(pins)}",
            )
        yield SourceNet(net_name, declared_degree, pins, source_line)


def normalize_ispd_degree_file(path: str | Path, design_id: str, degree: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    pins: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    scales: list[float] = []
    ids: list[bytes] = []
    invalid: dict[str, int] = {}
    parsed = 0
    for source_net in parse_degree_file(path):
        parsed += 1
        try:
            normalized, transform = normalize_net(source_net.pins, expected_degree=degree)
        except InvalidNetError as error:
            invalid[error.reason] = invalid.get(error.reason, 0) + 1
            continue
        pins.append(normalized.astype(np.float32))
        offsets.append(transform.offset_xy.astype(np.float64))
        scales.append(transform.scale)
        ids.append(f"ispd:{design_id}:{source_net.net_name}:d{degree}".encode("utf-8"))
    arrays = {
        "net_ids": np.asarray(ids, dtype="S128"),
        "offsets": np.asarray(offsets, dtype=np.float64).reshape((-1, 2)),
        "pins": np.asarray(pins, dtype=np.float32).reshape((-1, degree, 2)),
        "scales": np.asarray(scales, dtype=np.float64),
    }
    metadata: dict[str, object] = {
        "schema_version": 1,
        "kind": "ispd_protocol_a_inputs",
        "design_id": design_id,
        "degree": degree,
        "parsed_count": parsed,
        "valid_count": len(pins),
        "invalid_count": parsed - len(pins),
        "invalid_reasons": dict(sorted(invalid.items())),
        "source_path": str(Path(path)),
        "source_sha256": file_sha256(Path(path)),
        "contains_model_metrics": False,
    }
    return arrays, metadata
