"""A minimal BERT-shaped stub model.

The package's adapters target HuggingFace's `encoder.layer[i]` layout,
but nothing in `prunelib.transformer` needs HuggingFace itself -- only a
module tree with that shape. Building the tree by hand keeps the suite
fast, offline, and free of a `transformers`/`datasets` dependency, and
makes it trivial to construct pathological cases (dead neurons, exact
twins, fp16 weights) that a real checkpoint would never hand us.
"""
import pytest
import torch
import torch.nn as nn


class StubConfig:
    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size


class StubIntermediate(nn.Module):
    """Mirrors `BertIntermediate`: dense, then the activation."""

    def __init__(self, hidden, intermediate, bias=True):
        super().__init__()
        self.dense = nn.Linear(hidden, intermediate, bias=bias)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, x):
        return self.intermediate_act_fn(self.dense(x))


class StubFFNOutput(nn.Module):
    def __init__(self, intermediate, hidden, bias=True):
        super().__init__()
        self.dense = nn.Linear(intermediate, hidden, bias=bias)

    def forward(self, x):
        return self.dense(x)


class StubSelfAttention(nn.Module):
    """Mirrors `BertSelfAttention`, including its own `num_attention_heads`
    / `attention_head_size` / `all_head_size` -- attributes the *module*
    tracks separately from `config.num_attention_heads`, and the ones
    `attention_surgery.AttentionSurgeon` must update after removing heads.
    """

    def __init__(self, hidden, num_attention_heads):
        super().__init__()
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def _split_heads(self, x):
        return x.view(*x.shape[:-1], self.num_attention_heads, self.attention_head_size).permute(
            0, 2, 1, 3
        )

    def forward(self, x):
        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
        context = torch.matmul(torch.softmax(scores, dim=-1), v)
        return context.permute(0, 2, 1, 3).reshape(*x.shape[:-1], self.all_head_size)


class StubSelfOutput(nn.Module):
    """Mirrors `BertSelfOutput`: the attention block's output projection
    plus the residual add, which is why `attention_surgery` must shrink
    only its *input* columns -- its output width (and thus the residual
    it lands on) is the model's hidden size, unaffected by head count.
    """

    def __init__(self, hidden):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)

    def forward(self, x, residual):
        return self.dense(x) + residual


class StubAttention(nn.Module):
    def __init__(self, hidden, num_attention_heads):
        super().__init__()
        # Attribute is literally named `self` in HuggingFace's layout.
        setattr(self, "self", StubSelfAttention(hidden, num_attention_heads))
        self.output = StubSelfOutput(hidden)

    def forward(self, x):
        return self.output(getattr(self, "self")(x), x)


class StubLayer(nn.Module):
    def __init__(self, hidden, intermediate, num_attention_heads, bias=True):
        super().__init__()
        self.attention = StubAttention(hidden, num_attention_heads)
        self.intermediate = StubIntermediate(hidden, intermediate, bias=bias)
        self.output = StubFFNOutput(intermediate, hidden, bias=bias)

    def forward(self, x):
        x = self.attention(x)
        return self.output(self.intermediate(x)) + x


class StubEncoder(nn.Module):
    def __init__(self, num_layers, hidden, intermediate, num_attention_heads, bias=True):
        super().__init__()
        self.layer = nn.ModuleList(
            StubLayer(hidden, intermediate, num_attention_heads, bias=bias)
            for _ in range(num_layers)
        )


class StubBert(nn.Module):
    def __init__(self, vocab=40, hidden=8, intermediate=16, num_layers=3, heads=2, bias=True):
        super().__init__()
        self.config = StubConfig(hidden, heads, intermediate)
        self.embeddings = nn.Embedding(vocab, hidden)
        self.encoder = StubEncoder(num_layers, hidden, intermediate, heads, bias=bias)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask=None, **_):
        x = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            x = layer(x)
        return x


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(1234)


@pytest.fixture
def model():
    return StubBert()


@pytest.fixture
def batches():
    """Two batches with genuine padding, so masking is actually exercised."""
    return [
        {
            "input_ids": torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        },
        {
            "input_ids": torch.tensor([[8, 9, 10, 11]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        },
    ]
