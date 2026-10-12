"""Strict CircuitOps IR adapter, checked against the original final DEF.

Coordinates are upstream OpenDB ITerm.getAvgXY global geometry centers in DBU.
The upstream IR excludes top-level BTerms. We reject each entire affected net,
never reduce its degree. No training, degree selection, deduplication or scaling
other than conversion from DBU to microns happens in this adapter.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
from pathlib import Path
import re
from typing import Iterator

import numpy as np


def _text(path):
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(encoding="utf-8", newline="")


def _canonical(name):
    # DEF escaping and the upstream Tcl output both escape bus delimiters.
    return re.sub(r"\\(.)", r"\1", name)


def _descriptor(path):
    return dict(path) if isinstance(path, dict) else json.loads(Path(path).read_text())


def _def_metadata(path):
    data = {}
    with _text(path) as f:
        for line in f:
            m = re.match(r"\s*UNITS DISTANCE MICRONS (\d+)\s*;", line)
            if m:
                data["dbu_per_micron"] = int(m[1])
            m = re.match(r"\s*(COMPONENTS|PINS|SPECIALNETS|NETS)\s+(\d+)\s*;", line)
            if m:
                data[m[1].lower() + "_declared"] = int(m[2])
                if m[1] == "NETS":
                    break
    if not data.get("dbu_per_micron") or "nets_declared" not in data:
        raise ValueError(f"Missing DEF units or NETS section: {path}")
    return data


def _iter_def_nets(path):
    """Stream only NETS connectivity prefixes, ignoring all routing geometry."""
    active = False
    prefix = []
    connectivity_done = False
    with _text(path) as f:
        for line in f:
            line = line.strip()
            if not active:
                if re.fullmatch(r"NETS\s+\d+\s*;", line):
                    active = True
                continue
            if line == "END NETS":
                if prefix:
                    raise ValueError("Unterminated DEF net")
                return
            if not line or line.startswith("#"):
                continue
            if line.startswith("- "):
                if prefix:
                    raise ValueError("Nested/unterminated DEF net")
                prefix = []
                connectivity_done = False
            if not connectivity_done:
                # DEF properties begin at a whitespace-delimited +; escaped +
                # inside identifiers is retained. CONNECTED pins precede it.
                split = re.split(r"(?:^|\s+)\+\s+", line, maxsplit=1)
                prefix.append(split[0].split(";", 1)[0])
                connectivity_done = len(split) == 2
            if ";" in line:
                record = " ".join(prefix)
                m = re.match(r"-\s+(\S+)\s*(.*)", record)
                if not m:
                    raise ValueError(f"Cannot parse DEF net prefix: {record[:200]}")
                terms = re.findall(r"\(\s*(\S+)\s+(\S+)\s*\)", m[2])
                residue = re.sub(r"\(\s*(\S+)\s+(\S+)\s*\)", "", m[2]).strip()
                if residue:
                    raise ValueError(f"Unparsed DEF connectivity: {residue[:200]}")
                yield _canonical(m[1]), [(_canonical(a), _canonical(b)) for a, b in terms]
                prefix = []
                connectivity_done = False
    raise ValueError("Missing END NETS")


def iter_design_records(path) -> Iterator[dict]:
    """Yield every ordinary DEF net, including invalid records and reasons.

    Keys: net_id, raw_degree, points (float64 micron array or None), valid,
    reasons, ir_degree, missing_pins, extra_pins, is_clock.
    Raises on schema/identifier ambiguity or unexpected global inconsistency.
    """
    desc = _descriptor(path)
    meta = _def_metadata(desc["def_path"])
    units = meta["dbu_per_micron"]
    pins = {}
    by_net = collections.defaultdict(dict)
    bad_pin_reasons = collections.defaultdict(set)
    with _text(desc["pin_properties_path"]) as f:
        reader = csv.DictReader(f)
        required = {"pin_name", "net_name", "x", "y", "is_in_clk"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing required CircuitOps pin columns")
        for row in reader:
            pin, net = _canonical(row["pin_name"]), _canonical(row["net_name"])
            if pin in pins:
                raise ValueError(f"Duplicate/colliding pin identifier: {pin}")
            x, y = float(row["x"]), float(row["y"])
            if not math.isfinite(x) or not math.isfinite(y):
                bad_pin_reasons[net].add("nonfinite_coordinate")
            if x == -1 or y == -1:
                bad_pin_reasons[net].add("upstream_missing_coordinate_sentinel")
            pins[pin] = (net, x, y, row["is_in_clk"] == "1")
            by_net[net][pin] = pins[pin]
    properties = set()
    with _text(desc["net_properties_path"]) as f:
        reader = csv.DictReader(f)
        if "net_name" not in (reader.fieldnames or []):
            raise ValueError("Missing net_name property column")
        for row in reader:
            net = _canonical(row["net_name"])
            if net in properties:
                raise ValueError(f"Duplicate/colliding net property: {net}")
            properties.add(net)
    edges = collections.defaultdict(set)
    with _text(desc["net_pin_edge_path"]) as f:
        reader = csv.DictReader(f)
        if not {"src", "tar", "src_type", "tar_type"} <= set(reader.fieldnames or []):
            raise ValueError("Missing edge columns")
        for row in reader:
            if (row["src_type"], row["tar_type"]) == ("net", "pin"):
                net, pin = _canonical(row["src"]), _canonical(row["tar"])
            elif (row["src_type"], row["tar_type"]) == ("pin", "net"):
                pin, net = _canonical(row["src"]), _canonical(row["tar"])
            else:
                raise ValueError(f"Unsupported edge types: {row}")
            if pin in edges[net]:
                raise ValueError(f"Duplicate edge: {net} {pin}")
            edges[net].add(pin)
    seen = set()
    for net, terms in _iter_def_nets(desc["def_path"]):
        if net in seen:
            raise ValueError(f"Duplicate/colliding DEF net identifier: {net}")
        seen.add(net)
        reasons = set(bad_pin_reasons.get(net, ()))
        top_ports = [pin for inst, pin in terms if inst == "PIN"]
        expected = [f"{inst}/{pin}" for inst, pin in terms if inst != "PIN"]
        expect_set = set(expected)
        actual = by_net.get(net, {})
        if len(set(terms)) != len(terms):
            reasons.add("duplicate_DEF_terminal")
        if top_ports:
            reasons.add("missing_top_level_io_in_upstream_IR")
        missing = sorted(expect_set - actual.keys())
        extra = sorted(actual.keys() - expect_set)
        if missing:
            reasons.add("missing_instance_pin_in_upstream_IR")
        if extra:
            reasons.add("extra_instance_pin_in_upstream_IR")
        if net not in properties:
            reasons.add("missing_net_property")
        if edges.get(net, set()) != expect_set:
            reasons.add("edge_DEF_pin_set_mismatch")
        if not terms:
            reasons.add("empty_DEF_net")
        points = None
        if not reasons:
            # DEF pin order gives a stable geometry ordering; no pin is dropped.
            points = np.asarray([(actual[p][1], actual[p][2]) for p in expected], dtype=np.float64) / units
            if points.shape != (len(terms), 2):
                raise AssertionError("Degree changed during coordinate assembly")
        yield {"net_id": net, "raw_degree": len(terms), "ir_degree": len(actual),
               "points": points, "valid": not reasons, "reasons": sorted(reasons),
               "missing_pins": missing + ["PIN/" + p for p in top_ports], "extra_pins": extra,
               "is_clock": any(p[3] for p in actual.values())}
    if len(seen) != meta["nets_declared"]:
        raise ValueError(f"DEF declared {meta['nets_declared']} nets; parsed {len(seen)}")
    # Missing DEF nets must not become silently available through the IR.
    orphans = (set(by_net) | set(edges) | properties) - seen
    if orphans:
        raise ValueError(f"IR contains {len(orphans)} nets absent from DEF: {sorted(orphans)[:10]}")


def parse_design(path):
    """Compatibility generator. Use iter_design_records for exclusion auditing."""
    for record in iter_design_records(path):
        if record["valid"]:
            yield record["net_id"], record["points"]


def audit_design(path, output_dir=None):
    desc = _descriptor(path)
    counts = collections.Counter()
    raw_degrees = collections.Counter()
    valid_degrees = collections.Counter()
    rejected_degrees = collections.Counter()
    reasons = collections.Counter()
    output = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = (output_dir / (desc["design_id"] + "_rejected_nets.jsonl")).open("w")
    try:
        for record in iter_design_records(desc):
            counts["raw_nets"] += 1
            raw_degrees[record["raw_degree"]] += 1
            if record["valid"]:
                counts["complete_nets"] += 1
                valid_degrees[record["raw_degree"]] += 1
            else:
                counts["rejected_nets"] += 1
                rejected_degrees[record["raw_degree"]] += 1
                reasons.update(record["reasons"])
                if output:
                    output.write(json.dumps({k: v for k, v in record.items() if k != "points"}) + "\n")
    finally:
        if output:
            output.close()
    result = {"design_id": desc["design_id"], "metadata": _def_metadata(desc["def_path"]),
              **dict(counts), "raw_degree_counts": dict(sorted(raw_degrees.items())),
              "complete_degree_counts": dict(sorted(valid_degrees.items())),
              "rejected_degree_counts": dict(sorted(rejected_degrees.items())),
              "rejection_reason_counts": dict(reasons),
              "target_degree_complete_counts": {d: valid_degrees[d] for d in (10, 20, 30, 40, 50)},
              "audit_pass": True}
    if output_dir:
        (output_dir / (desc["design_id"] + "_audit.json")).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("descriptor")
    ap.add_argument("--audit-dir")
    args = ap.parse_args()
    print(json.dumps(audit_design(args.descriptor, args.audit_dir), indent=2))
