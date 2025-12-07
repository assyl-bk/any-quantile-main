import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicityLoss(nn.Module):
    """
    Penalize quantile crossing during training.
    When predicting multiple quantiles simultaneously, higher quantile levels
    should predict higher values. This loss penalizes violations of this constraint.
    """
    def __init__(self, margin: float = 0.0, reduction: str = 'mean'):
        """
        Args:
            margin: Minimum gap between adjacent quantile predictions
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, quantiles: torch.Tensor):
        """
        Args:
            predictions: [B, H, Q] - predicted values for Q quantiles
            quantiles: [B, 1, Q] or [B, Q] - quantile levels (should be sorted)
        
        Returns:
            Scalar loss penalizing monotonicity violations
        """
        # Ensure quantiles are 3D
        if quantiles.dim() == 2:
            quantiles = quantiles.unsqueeze(1)  # [B, Q] -> [B, 1, Q]
        
        # Sort quantiles and get sorting indices
        q_sorted, sort_idx = quantiles.sort(dim=-1)
        
        # Apply same sorting to predictions
        sort_idx_expanded = sort_idx.expand_as(predictions)
        pred_sorted = predictions.gather(-1, sort_idx_expanded)
        
        # Compute differences between adjacent quantile predictions
        # For proper ordering: pred[q_i+1] >= pred[q_i] + margin
        diffs = pred_sorted[..., 1:] - pred_sorted[..., :-1]  # [B, H, Q-1]
        
        # Penalize negative differences (crossings)
        # Using soft penalty: ReLU(-diff + margin)
        violations = F.relu(-diffs + self.margin)
        
        # Apply reduction
        if self.reduction == 'mean':
            return violations.mean()
        elif self.reduction == 'sum':
            return violations.sum()
        else:
            return violations