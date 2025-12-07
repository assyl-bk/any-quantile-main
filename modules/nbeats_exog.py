import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# EXOGENOUS FEATURE ENCODER
# ============================================================

class ExogenousEncoder(nn.Module):
    """Encode continuous + calendar exogenous features into embedding space."""

    def __init__(
        self,
        num_continuous: int = 4,    # temperature, humidity, etc.
        num_calendar: int = 4,      # hour, dow, month, is_weekend
        embed_dim: int = 64
    ):
        super().__init__()

        # Continuous features (weather) projection
        self.continuous_proj = nn.Sequential(
            nn.Linear(num_continuous, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Calendar feature embeddings (each uses 1/4 of embed_dim)
        chunk = embed_dim // 4

        self.hour_embed = nn.Embedding(24, chunk)
        self.dow_embed = nn.Embedding(7, chunk)
        self.month_embed = nn.Embedding(12, chunk)
        self.weekend_embed = nn.Embedding(2, chunk)

    def forward(self, continuous, calendar):
        """
        Args:
            continuous: [B, T, num_continuous]
            calendar: [B, T, 4] (hour, dow, month, is_weekend) as integer indices

        Returns:
            [B, T, embed_dim] combined embedding
        """

        # Project continuous features
        cont_embed = self.continuous_proj(continuous)

        # Calendar embeddings
        cal_embed = torch.cat([
            self.hour_embed(calendar[..., 0].long()),
            self.dow_embed(calendar[..., 1].long()),
            self.month_embed(calendar[..., 2].long()),
            self.weekend_embed(calendar[..., 3].long())
        ], dim=-1)

        return cont_embed + cal_embed



# ============================================================
# N-BEATS BLOCK WITH EXOGENOUS CONDITIONING
# ============================================================

class NbeatsBlockWithExog(NbeatsBlockConditioned):
    """N-BEATS block extended with exogenous feature conditioning."""

    def __init__(
        self,
        num_layers: int,
        layer_width: int,
        size_in: int,
        size_out: int,
        exog_dim: int = 64
    ):
        super().__init__(num_layers, layer_width, size_in, size_out)

        # Project exogenous feature embedding to match layer width
        self.exog_projection = nn.Linear(exog_dim, layer_width)

        # Gate to control how much exogenous info is injected
        self.exog_gate = nn.Sequential(
            nn.Linear(layer_width * 2, layer_width),
            nn.Sigmoid()
        )

    def forward(self, x, condition, exog_embed):
        """
        Args:
            x: [B, Q, layer_width] input series (Q = quantiles)
            condition: [B, Q, layer_width] quantile conditioning
            exog_embed: [B, T, exog_dim] exogenous feature embeddings
        """

        h = x

        # Average exogenous info through time
        exog_pooled = self.exog_projection(exog_embed.mean(dim=1))  # [B, layer_width]
        exog_pooled = exog_pooled.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, Q, layer_width]

        for i, layer in enumerate(self.fc_layers):
            h = F.relu(layer(h))

            if i == 0:
                # FiLM conditioning
                cond = self.condition_film(condition)
                offset, delta = cond[..., :self.layer_width], cond[..., self.layer_width:]
                h = h * (1 + delta) + offset

            # Apply exogenous gating
            gate = self.exog_gate(torch.cat([h, exog_pooled], dim=-1))
            h = h + gate * exog_pooled

        return self.forward_projection(h), self.backward_projection(h)
