"""Shared per-net call accounting used by training, search, and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class CallLedger:
    """Counts evaluated nets, never batch-function invocations."""

    actor_forward_nets: int = 0
    surrogate_evaluator_nets: int = 0
    exact_rest_evaluator_nets: int = 0
    geosteiner_nets: int = 0

    def add(self, field: str, count: int) -> None:
        if field not in self.__dataclass_fields__:
            raise KeyError(f"unknown call-ledger field: {field}")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("call count must be a nonnegative integer")
        setattr(self, field, getattr(self, field) + count)

    def merge(self, other: "CallLedger") -> None:
        for field in self.__dataclass_fields__:
            self.add(field, getattr(other, field))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

