"""Parse official ISPD 2007/2008 global-routing inputs into original pin XY.

`parse_design(path)` yields `(net_id, float64 array [number_of_pins, 2])`.
No pin sorting, deduplication, degree change, or normalization occurs. Layers
are projected out because REST consumes a 2-D point set; repeated XY remain.
Physical pin coordinates are used directly, never quantized to routing tiles.
"""
from __future__ import annotations
import gzip
from pathlib import Path
from typing import Iterator
import numpy as np


def _open(path):
    return gzip.open(path, 'rt', encoding='ascii') if str(path).endswith('.gz') else open(path, 'r', encoding='ascii')


def parse_design(path: str | Path) -> Iterator[tuple[str, np.ndarray]]:
    """Read every declared net, strictly validating finite pin coordinates."""
    with _open(path) as stream:
        first = stream.readline().split()
        if len(first) != 4 or first[0] != 'grid':
            raise ValueError(f'{path}: missing ISPD grid header')
        layers = int(first[3])
        for line in stream:
            fields = line.split()
            if fields[:2] == ['num', 'net']:
                if len(fields) != 3:
                    raise ValueError(f'{path}: malformed num net header')
                n_nets = int(fields[2])
                break
        else:
            raise ValueError(f'{path}: missing num net header')
        for index in range(n_nets):
            line = stream.readline()
            while line and not line.strip():
                line = stream.readline()
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f'{path}: malformed net header at net {index}: {line!r}')
            net_name, net_index, degree_s, width_s = fields
            degree = int(degree_s)
            if degree < 0:
                raise ValueError(f'{path}: negative degree at net {index}')
            xy = np.empty((degree, 2), dtype=np.float64)
            for pin in range(degree):
                coords = stream.readline().split()
                if len(coords) != 3:
                    raise ValueError(f'{path}: malformed pin {pin} of net {index}')
                xy[pin, 0], xy[pin, 1] = float(coords[0]), float(coords[1])
                layer = int(coords[2])
                if not 1 <= layer <= layers:
                    raise ValueError(f'{path}: invalid layer {layer} at net {index}')
            if not np.isfinite(xy).all():
                raise ValueError(f'{path}: nonfinite coordinates at net {index}')
            yield f'{net_name}::{net_index}', xy
