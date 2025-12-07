import numpy as np
import torch


class AdaptiveQuantileSampler:
    """
    Sample quantiles with probability proportional to their training loss.
    
    Maintains a running estimate of per-quantile loss and samples more
    frequently from quantiles where the model struggles.
    """
    def __init__(self,
                 num_bins: int = 100,
                 momentum: float = 0.99,
                 temperature: float = 1.0,
                 min_prob: float = 0.001):
        """
        Args:
            num_bins: Number of quantile bins to track
            momentum: EMA momentum for loss estimates (higher = slower adaptation)
            temperature: Softmax temperature for sampling (higher = more uniform)
            min_prob: Minimum sampling probability per bin
        """
        self.num_bins = num_bins
        self.momentum = momentum
        self.temperature = temperature
        self.min_prob = min_prob
        
        # Running loss estimates per quantile bin
        self.loss_estimates = np.ones(num_bins)
        self.bin_edges = np.linspace(0, 1, num_bins + 1)
        
        # Track number of updates per bin for better initialization
        self.bin_counts = np.zeros(num_bins)
    
    def update(self, quantiles: torch.Tensor, losses: torch.Tensor):
        """
        Update loss estimates with new observations.
        
        Args:
            quantiles: [B] or [B, Q] tensor of quantile values
            losses: [B] or [B, Q] tensor of corresponding losses
        """
        # Flatten to 1D
        quantiles_np = quantiles.detach().cpu().numpy().flatten()
        losses_np = losses.detach().cpu().numpy().flatten()
        
        # Filter out invalid values
        valid_mask = (~np.isnan(losses_np)) & (~np.isinf(losses_np))
        quantiles_np = quantiles_np[valid_mask]
        losses_np = losses_np[valid_mask]
        
        if len(quantiles_np) == 0:
            return
        
        # Bin the quantiles
        bin_indices = np.digitize(quantiles_np, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        # Update running estimates with exponential moving average
        for bin_idx, loss in zip(bin_indices, losses_np):
            # Use lower momentum for bins with fewer observations
            if self.bin_counts[bin_idx] < 10:
                effective_momentum = 0.5
            else:
                effective_momentum = self.momentum
            
            self.loss_estimates[bin_idx] = (
                effective_momentum * self.loss_estimates[bin_idx] +
                (1 - effective_momentum) * loss
            )
            self.bin_counts[bin_idx] += 1
    
    def sample(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """
        Sample quantiles with probability proportional to loss.
        
        Args:
            batch_size: Number of quantiles to sample
            device: Device to place the output tensor on
        
        Returns:
            [batch_size] tensor of sampled quantile values
        """
        # Softmax with temperature to get sampling probabilities
        probs = np.exp(self.loss_estimates / self.temperature)
        
        # Ensure minimum probability for exploration
        probs = np.maximum(probs, self.min_prob)
        
        # Normalize to get valid probability distribution
        probs = probs / probs.sum()
        
        # Sample bins according to probabilities
        bins = np.random.choice(self.num_bins, size=batch_size, p=probs)
        
        # Sample uniformly within each selected bin
        quantiles = (
            self.bin_edges[bins] +
            np.random.rand(batch_size) * (self.bin_edges[bins + 1] - self.bin_edges[bins])
        )
        
        # Clip to valid quantile range
        quantiles = np.clip(quantiles, 0.001, 0.999)
        
        # Convert to tensor
        quantiles_tensor = torch.tensor(quantiles, dtype=torch.float32)
        
        if device is not None:
            quantiles_tensor = quantiles_tensor.to(device)
        
        return quantiles_tensor
    
    def get_statistics(self):
        """Get statistics about the current sampling distribution."""
        probs = np.exp(self.loss_estimates / self.temperature)
        probs = np.maximum(probs, self.min_prob)
        probs = probs / probs.sum()
        
        return {
            'mean_loss': self.loss_estimates.mean(),
            'max_loss': self.loss_estimates.max(),
            'min_loss': self.loss_estimates.min(),
            'loss_std': self.loss_estimates.std(),
            'entropy': -(probs * np.log(probs + 1e-10)).sum(),
            'max_prob': probs.max(),
            'min_prob': probs.min(),
            'bins_used': (self.bin_counts > 0).sum(),
        }
    
    def reset(self):
        """Reset the sampler to initial state."""
        self.loss_estimates = np.ones(self.num_bins)
        self.bin_counts = np.zeros(self.num_bins)