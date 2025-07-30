import torch
import torch.nn as nn


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        # Create a table of embeddings for relative positions
        # It will account for negative, zero, and positive relative positions
        # num_units is the dimension of the embeddings
        # max_relative_position is the maximum distance for which we want to compute embeddings
        # In the sentence "This is awesome", max_relative_position" is 2
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        # Create a matrix of relative positions
        # The matrix will have shape (length_q, length_k)
        # where each element (i, j) is the relative position of the i-th query
        # and the j-th key. The relative position is computed as j - i
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        # Use the final_mat to index into the embeddings_table
        # This will give us the embeddings for the relative positions
        embeddings = self.embeddings_table[final_mat].cuda() 

        return embeddings

class RelPosMultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout = 0, device = 'cuda'):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 3  # Can be adjusted based on the application

        # We then learn relative position embeddings for keys and values
        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query) # [batch size, query len, hid dim]
        key = self.fc_k(key) # [batch size, key len, hid dim]
        value = self.fc_v(value) # [batch size, key len, hid dim]

        # Reshape to [batch size, query len, n heads, head dim]
        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim = -1)) # [batch size, n heads, query len, key len]

        # Compute the attention output
        # r_v1 is the reshaped value tensor
        # r_v2 is the relative position embeddings for values
        # weight1 is the attention output for the reshaped value tensor
        # weight2 is the attention output for the relative position embeddings
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2 # [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous() # [batch size, query len, n heads, head dim]
                
        x = x.view(batch_size, -1, self.hid_dim) # [batch size, query len, hid dim]
                
        x = self.fc_o(x) # [batch size, query len, hid dim]
        
        return x


class TransformerEncoderBlock(nn.Module):
  def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1):
    super().__init__()
    self.attention = RelPosMultiHeadAttention(embed_dim, num_heads, dropout)
    self.ff = nn.Sequential(nn.Linear(embed_dim, ff_hidden_dim),
                            nn.ReLU(),
                            nn.Linear(ff_hidden_dim, embed_dim))
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, X: torch.Tensor) -> torch.Tensor:
    # X shape: [T, B, E] where T = num_slices, B = batch, E = embed_dim
    attn_out = self.attention(X, X, X)
    X = self.norm1(X + self.dropout(attn_out))
    ff_out = self.ff(X)
    return self.norm2(X + self.dropout(ff_out))


class TransformerEncoder(nn.Module):
  def __init__(self, num_layers: int, embed_dim: int, num_heads: int,
               ff_hiden_dim: int, dropout: float = 0.1):
    super().__init__()
    self.layers = nn.ModuleList([
      TransformerEncoderBlock(embed_dim, num_heads, ff_hiden_dim) for _ in range(num_layers)
    ])
  
  def forward(self, X: torch.Tensor) -> torch.Tensor:
    # X shape: [T, B, E] where T = num_slices, B = batch, E = embed_dim
    for layer in self.layers:
      X = layer(X)
    return X