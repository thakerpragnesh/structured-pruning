"""
Name -> implementation registries: the one extension mechanism every
pluggable part of `prunelib` shares -- saliency scorers (`saliency.py`),
selection rules (`selection.py`), distance metrics (`distance.py`), and
quantization methods (`quantization.py`). Per-module-type pruning rules
(`module_rules.py`) are keyed by class instead of by name, and resolved
through the MRO with `resolve_by_type` below.

Before this module, each of those lived in a private dict
(`saliency._METHODS`, `quantization._MODEL_METHODS`) or an `if metric ==
...` chain (`clustering._distance`, `scanners.pairwise_distance_matrix`, the
same three metrics written out twice), so adding a method meant editing the
library itself -- KT.md section 7 used to say exactly that ("register it in
`_METHODS`"). A registry is open for extension from the caller's own code
(`register_saliency_method("taylor", fn)`) and closed for modification:
the dispatch code that reads it never changes.
"""
from __future__ import annotations

from typing import Generic, Mapping, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Named implementations of one kind of thing (`kind` is only used in
    error messages: "unknown {kind} 'x', expected one of [...]")."""

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str, entry: T | None = None, *, overwrite: bool = False):
        """Register `entry` under `name`, or -- called without `entry` --
        return a decorator that does. Re-registering an existing name raises
        unless `overwrite=True`, so two plugins can't silently shadow each
        other (or a built-in)."""

        def add(e: T) -> T:
            if name in self._entries and not overwrite:
                raise ValueError(f"{self.kind} {name!r} is already registered; pass overwrite=True to replace it")
            self._entries[name] = e
            return e

        return add if entry is None else add(entry)

    def unregister(self, name: str) -> None:
        self.get(name)  # raises the usual "unknown ..." error for a missing name
        del self._entries[name]

    def get(self, name: str) -> T:
        try:
            return self._entries[name]
        except KeyError:
            raise ValueError(f"unknown {self.kind} {name!r}, expected one of {self.names()}") from None

    def resolve(self, name_or_entry: str | T) -> T:
        """A registered name is looked up; anything else is taken to already
        be an implementation and returned as-is. This is what lets every
        `method=`/`metric=` argument in `prunelib` accept either a name or
        the implementation itself, without each call site re-deciding."""
        return self.get(name_or_entry) if isinstance(name_or_entry, str) else name_or_entry

    def names(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries


def resolve_by_type(entries: Mapping[type, T], cls: type) -> T | None:
    """The entry for `cls` or its nearest registered base class, else None.

    Walking the MRO gives the same answer an `isinstance` chain would -- a
    subclass of a registered type (`nn.LazyConv2d`, a user's `nn.Linear`
    subclass) gets its parent's entry -- without the chain having to list
    every type it knows about."""
    for base in cls.__mro__:
        if base in entries:
            return entries[base]
    return None

