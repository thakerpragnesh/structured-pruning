"""Physically resizing modules once a criterion has decided -- mechanics, not
policy.

- `linear_ops` -- `rebuild_linear`: the one way a resized `nn.Linear` is built.
- `ffn_surgery` -- `FFNSurgeon`: slice an FFN pair, fold merges, compensate
  the bias.
- `attention_surgery` -- `AttentionSurgeon`: remove whole heads, optionally
  mean-compensated.

Depends on: `selection` (only for the `Selection` type `FFNSurgeon` consumes).
"""
