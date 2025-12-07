import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QuantileConditionedAttention(nn.Module):
    """
    Multi-head attention where queries are conditioned on quantile level.
    This allows different quantiles to attend to different parts of history:
    - Extreme quantiles (0.01, 0.99) may focus on peak/trough events
    - Central quantiles (0.5) may attend more uniformly
    """
    def __init__(self,
                 d_model: int = 256,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Standard Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Quantile conditioning for queries
        self.quantile_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, quantile):
        """
        Args:
            x: [B, T, d_model] - input sequence
            quantile: [B, 1] or [B, Q] - quantile levels
        
        Returns:
            [B, T, d_model] - attended output
        """
        B, T, _ = x.shape
        
        # Handle quantile dimensions
        if quantile.dim() == 2:
            quantile = quantile.unsqueeze(-1)  # [B, Q] -> [B, Q, 1]
        
        # Get quantile conditioning
        q_embed = self.quantile_proj(quantile)  # [B, Q, d_model]
        
        # If we have multiple quantiles, we need to expand x
        if q_embed.shape[1] > 1:
            Q_dim = q_embed.shape[1]
            x_expanded = x.unsqueeze(1).expand(-1, Q_dim, -1, -1)  # [B, Q, T, d_model]
            B, Q, T, _ = x_expanded.shape
            x_reshaped = x_expanded.reshape(B * Q, T, self.d_model)
            q_embed_reshaped = q_embed.reshape(B * Q, 1, self.d_model)
        else:
            x_reshaped = x
            q_embed_reshaped = q_embed
            Q = 1
        
        # Compute Q, K, V
        Q_proj = self.W_q(x_reshaped) + q_embed_reshaped  # Condition queries on quantile
        K_proj = self.W_k(x_reshaped)
        V_proj = self.W_v(x_reshaped)
        
        # Reshape for multi-head attention
        batch_size = Q_proj.shape[0]
        seq_len = Q_proj.shape[1]
        
        Q_proj = Q_proj.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K_proj = K_proj.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V_proj = V_proj.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention and reshape
        out = torch.matmul(attn, V_proj)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection + residual + layer norm
        out = self.layer_norm(x_reshaped + self.W_o(out))
        
        # Reshape back if we had multiple quantiles
        if Q > 1:
            out = out.view(B, Q, T, self.d_model)
        
        return out