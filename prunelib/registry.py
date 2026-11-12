"""
Name -> implementation registries: the one extension mechanism every
pluggable part of `prunelib` shares -- saliency scorers (`saliency.py`),
selection rules (`selection.py`), distance metrics (`distance.py`),
quantization methods (`quantization.py`), per-layer-type pruning rules
(`module_rules.py`) and the ops a prune passes through (`op_rules.py`). The
last two are keyed by class (or fx op target) instead of by name, and looked
up through the MRO with `resolve_by_type` below.

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

from types import MappingProxyType
from typing import Generic, Hashable, Mapping, TypeVar

T = TypeVar("T")


def _label(key: Hashable) -> str:
    """How a key reads in an error message: a name quoted, a class or
    function by its name (`Conv2d`, not `<class 'torch.nn...Conv2d'>`)."""
    return repr(key) if isinstance(key, str) else getattr(key, "__qualname__", repr(key))


class Registry(Generic[T]):
    """Implementations of one kind of thing, keyed by name -- or by class,
    for per-type entries, which are then looked up with `resolve_by_type(
    registry.entries(), cls)`. `kind` is only used in error messages:
    "unknown {kind} 'x', expected one of [...]"."""

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[Hashable, T] = {}

    def register(self, name: Hashable, entry: T | None = None, *, overwrite: bool = False):
        """Register `entry` under `name`, or -- called without `entry` --
        return a decorator that does. Re-registering an existing name raises
        unless `overwrite=True`, so two plugins can't silently shadow each
        other (or a built-in)."""

        def add(e: T) -> T:
            if name in self._entries and not overwrite:
                raise ValueError(f"{self.kind} {_label(name)} is already registered; pass overwrite=True to replace it")
            self._entries[name] = e
            return e

        return add if entry is None else add(entry)

    def unregister(self, name: Hashable) -> None:
        self.get(name)  # raises the usual "unknown ..." error for a missing name
        del self._entries[name]

    def get(self, name: Hashable) -> T:
        try:
            return self._entries[name]
        except KeyError:
            raise ValueError(f"unknown {self.kind} {_label(name)}, expected one of {self.names()}") from None

    def resolve(self, name_or_entry: str | T) -> T:
        """A registered name is looked up; anything else is taken to already
        be an implementation and returned as-is. This is what lets every
        `method=`/`metric=` argument in `prunelib` accept either a name or
        the implementation itself, without each call site re-deciding."""
        return self.get(name_or_entry) if isinstance(name_or_entry, str) else name_or_entry

    def names(self) -> list:
        return list(self._entries)

    def entries(self) -> Mapping[Hashable, T]:
        """A read-only, live view of every entry -- what `resolve_by_type`
        searches, and what per-call overrides are layered over
        (`ChainMap(overrides, registry.entries())`)."""
        return MappingProxyType(self._entries)

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

