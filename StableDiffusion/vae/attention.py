import torch
import torch.nn as nn
import torch.nn.functional as F
import math




class TimeEmbedding(nn.Module):
  def __init__(self, num_timesteps, embedding_dim):
    super().__init__()
    self.num_timesteps = num_timesteps
    self.embedding_dim = embedding_dim
    pe = torch.zeros(self.num_timesteps, self.embedding_dim)
    posi = torch.arange(0, self.embedding_dim, 2)
    num = torch.arange(0, self.num_timesteps).unsqueeze(1)
    # print(num.shape)
    denm = 1 / 10000 ** (posi /self.embedding_dim)
    # print(denm.shape)
    pe[:, ::2] = torch.sin(num * denm)
    pe[:, 1::2] = torch.cos(num * denm)
    self.register_buffer('pe', pe)   ### pe become attribute and shape is (num_timesteps, embedding_dim)

  def forward(self, t):
    return self.pe[t]

  

class MultiHeadAttention_Images(nn.Module):
  def __init__(self, channels, num_heads):
    super().__init__()
    self.norm = nn.GroupNorm(32, channels)
    self.w_q = nn.Conv2d(channels, channels, kernel_size = 1)
    self.w_k = nn.Conv2d(channels, channels, kernel_size = 1)
    self.w_v = nn.Conv2d(channels, channels, kernel_size = 1)
    self.proj = nn.Conv2d(channels, channels, kernel_size = 1)
    if channels % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
    self.head_dim = channels//num_heads
    self.num_heads = num_heads
  
  def forward(self, x):
    B, C, H, W = x.shape
    h = self.norm(x)
    q = self.w_q(h)
    k = self.w_k(h)
    v = self.w_v(h)
    q = q.reshape(B, self.num_heads, self.head_dim, H*W)
    k = k.reshape(B, self.num_heads, self.head_dim, H*W)
    v = v.reshape(B, self.num_heads, self.head_dim, H*W)
    
    attn = q.transpose(-2, -1) @ k   ### (B, h, H*W, H*W)
    attn = attn / math.sqrt(self.head_dim)
    attn = torch.nn.functional.softmax(attn, dim=-1)

    h = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  ###(B, h, T, H*W)
    h = h.reshape(B, C, H, W)
    return self.proj(h) + x






class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()   ### C --> d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = d_model // num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.w_o = nn.Linear(d_model, d_model)


    def forward(self, x, casual_mask = False):
        B, T, C = x.shape
        query = self.w_q(x)  ### (B, T, C)
        key   = self.w_k(x)  ### (B, T, C)
        value = self.w_v(x)   ### (B, T, C)
        query = query.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  ### (B, H, T, d)
        key   = key.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  ### (B, H, T, d)
        value = value.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  ###(B, H, T, d)
        scores = query @ key.transpose(-2, -1)   #####(B, H, T, T)

        if casual_mask:
            mask = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(scores / math.sqrt(self.head_dim)) ###(B, H, T, T)
        attn = self.dropout(attn)
        out = attn @ value   ### (B, H, T, d)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)  # (B, T, C)
        out = self.w_o(out)
        return out






class CrossAttention(nn.Module):
  def __init__(self, n_head, d_emb, d_cross):
    ###d_emb = T, d_cross = T of other
    super().__init__()
    self.w_q = nn.Linear(d_emb, d_emb)
    self.w_k = nn.Linear(d_cross, d_emb)
    self.w_v = nn.Linear(d_cross, d_emb)
    self.out_proj = nn.Linear(d_emb, d_emb)
    self.n_head = n_head
    if d_emb % n_head != 0:
      raise ValueError("d_model must be divisible by num_heads")
    self.d_head = d_emb // n_head

  def forward(self, x, context):
    B, T, C = x.shape
    B, S, d_cross = context.shape

    q = self.w_q(x)   ### (B, T, C)  -> (B, T, C)
    k = self.w_k(context)   ### (B, S, d_cross)  -> (B, S, C)
    v = self.w_v(context)   ### (B, S, d_cross)  -> (B, S, C)

    q = q.reshape(B, T, self.n_head, self.d_head)   ###(B, T, n_head, d_head)
    q = q.permute(0, 2, 1, 3)    ### (B, n_head, T, d_head)

    k = k.reshape(B, S, self.n_head, self.d_head)    ### (B, S, n_head, d_head)
    k = k.permute(0, 2, 1, 3)   ### (B, n_head, S, d_head)

    v = v.reshape(B, S, self.n_head, self.d_head)   ### (B, S, n_head, d_head)
    v = v.permute(0, 2, 1, 3)    ### (B, n_head, S, d_head)

   
    attn = (q @ k.transpose(-2, -1)) ### (B, n_head, T, T)
    attn = attn * (1.0 / math.sqrt(k.size(-1)))

    attn = F.softmax(attn, dim=-1)

    y = attn @ v

    y = y.transpose(1, 2).reshape(B, T, C)
    y = self.out_proj(y)
    return y

