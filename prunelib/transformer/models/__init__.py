"""How to reach into a model: which layers exist, and how to get and set a
layer's FFN pair or attention quartet.

- `adapters` -- `FFNLayerAdapter` / `AttentionLayerAdapter` (and their shared
  `LayerStackAdapter` base), `BertLayerAdapter`, and `resolve_adapter` /
  `register_adapter`, the one place a consumer's adapter is chosen.
- `layers` -- `discover_layers`: find prunable `Conv2d`/`Linear` modules by
  type and name.

Depends on: nothing else in the package. Everything model-family-specific
lives here, so nothing outside this subpackage names a concrete layout.
"""
