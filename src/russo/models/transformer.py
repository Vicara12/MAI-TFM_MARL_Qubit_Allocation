import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
  def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
    super().__init__()
    # We use batch_first=True to have input shape [B, T, E] 
    self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    self.ff = nn.Sequential(nn.Linear(embed_dim, ff_hidden_dim),
                            nn.ReLU(),
                            nn.Linear(ff_hidden_dim, embed_dim))
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, X: torch.Tensor) -> torch.Tensor:
    # X shape: [B, T, E] where B = batch size, T = number of slices, E = embed_dim
    attn_out, _ = self.attention(X, X, X)
    X = self.norm1(X + self.dropout(attn_out))
    ff_out = self.ff(X)
    return self.norm2(X + self.dropout(ff_out))


class TransformerEncoder(nn.Module):
  def __init__(self, num_layers: int, embed_dim: int, num_heads: int,
               ff_hidden_dim: int, dropout: float = 0.1):
    super().__init__()
    self.layers = nn.ModuleList([
      TransformerEncoderBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)
    ])
  
  def forward(self, X: torch.Tensor) -> torch.Tensor:
    # X shape: [B, T, E] where B = batch size, T = num slices, E = embed_dim
    for layer in self.layers:
      X = layer(X)
    return X