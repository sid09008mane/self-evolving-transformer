import torch
import torch.nn as nn


# ------------------------------------------------
# Transformer Block
# ------------------------------------------------

class TransformerBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads, ff_dim):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, hidden_dim)
        )

        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):

        attn_output, _ = self.attention(x, x, x)

        x = self.norm1(x + attn_output)

        ff_output = self.ff(x)

        x = self.norm2(x + ff_output)

        return x


# ------------------------------------------------
# Dynamic Transformer
# ------------------------------------------------

class DynamicTransformer(nn.Module):

    def __init__(self, genome, vocab_size=50000, max_seq_len=128):
        super().__init__()

        hidden_dim = genome.hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        self.position_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, hidden_dim)
        )

        self.layers = nn.ModuleList()

        for _ in range(genome.num_layers):

            block = TransformerBlock(
                hidden_dim,
                genome.num_heads,
                genome.ff_dim
            )

            self.layers.append(block)

        self.norm = nn.LayerNorm(hidden_dim)

        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):

        seq_len = x.size(1)

        x = self.token_embedding(x)

        x = x + self.position_embedding[:, :seq_len, :]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        logits = self.output_head(x)

        return logits


# ------------------------------------------------
# Model Builder
# ------------------------------------------------

class ModelBuilder:

    def __init__(self, vocab_size=50000, max_seq_len=128):

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def build(self, genome):

        model = DynamicTransformer(
            genome,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len
        )

        return model