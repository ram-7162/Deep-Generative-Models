import torch
import torch.nn as nn
import torch.nn.functional as f
from .attention import MultiHeadAttention




class ClipEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size, max_seq_len):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

    def forward(self, x):
        x = self.input_embedding(x)
        x = x + self.position_embedding
        return x




class ClipLayer(nn.Module):
    def __init__(self, n_head, d_model):
        super().__init__()
        self.attention = MultiHeadAttention(n_head, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        residue = x
        x = self.attention(x, casual_mask = True)
        x = self.norm1(x + residue)
        residue = x
        x = self.mlp(x)
        x = self.norm2(x + residue)
        return x




class Clip(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_embedding = ClipEmbedding(d_model=512, vocab_size=49408, max_seq_len=77)
        self.clip_layer = nn.ModuleList([ClipLayer(n_head=8, d_model=512) for _ in range(12)])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, x):
        x = x.type(torch.long)
        embed_token = self.clip_embedding(x)
        for layer in self.clip_layer:
            embed_token = layer(embed_token)
        return self.layernorm(embed_token)

    