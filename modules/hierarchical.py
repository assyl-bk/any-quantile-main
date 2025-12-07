import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalQuantilePredictor(nn.Module):
    """
    Two-stage hierarchical quantile prediction:
    Stage 1: Predict location (median) and scale (IQR or std)
    Stage 2: Predict quantile offsets conditioned on location/scale
    
    This ensures extreme quantiles are coherent with central tendency.
    """
    def __init__(self, backbone, size_in, size_out, layer_width):
        super().__init__()
        self.backbone = backbone
        self.size_in = size_in
        self.size_out = size_out
        self.layer_width = layer_width
        
        # Stage 1: Predict median and scale
        self.location_head = nn.Sequential(
            nn.Linear(layer_width, layer_width // 2),
            nn.ReLU(),
            nn.Linear(layer_width // 2, size_out)  # Median prediction
        )
        
        self.scale_head = nn.Sequential(
            nn.Linear(layer_width, layer_width // 2),
            nn.ReLU(),
            nn.Linear(layer_width // 2, size_out),
            nn.Softplus()  # Scale must be positive
        )
        
        # Stage 2: Quantile offset predictor
        self.offset_net = nn.Sequential(
            nn.Linear(layer_width + 1, layer_width),  # +1 for quantile
            nn.ReLU(),
            nn.Linear(layer_width, size_out)
        )
    
    def encode(self, x):
        """
        Extract features from input using backbone.
        Can be overridden if backbone has a different interface.
        """
        if hasattr(self.backbone, 'encode'):
            return self.backbone.encode(x)
        else:
            # If backbone doesn't have encode method, use forward pass
            # and extract intermediate features
            return self._extract_features(x)
    
    def _extract_features(self, x):
        """
        Fallback method to extract features when backbone doesn't have encode().
        Assumes backbone is an MLP or similar architecture.
        """
        if isinstance(self.backbone, nn.Sequential):
            # For sequential models, pass through all but last layer
            h = x
            for layer in list(self.backbone.children())[:-1]:
                h = layer(h)
            return h
        else:
            # For custom backbones, assume forward returns features
            # You may need to modify this based on your backbone architecture
            return self.backbone(x)
    
    def forward(self, x, q):
        """
        Args:
            x: [B, T] - input history
            q: [B, Q] - quantile levels to predict
        
        Returns:
            [B, H, Q] - quantile predictions
        """
        B = x.shape[0]
        Q = q.shape[-1]
        
        # Get backbone features
        features = self.encode(x)  # [B, layer_width]
        
        # Ensure features have correct shape
        if features.dim() > 2:
            features = features.view(B, -1)
        
        # Stage 1: Location and scale
        median = self.location_head(features)  # [B, H]
        scale = self.scale_head(features)  # [B, H]
        
        # Stage 2: Quantile-specific offsets
        features_expanded = features.unsqueeze(1).expand(-1, Q, -1)  # [B, Q, W]
        q_expanded = q.unsqueeze(-1)  # [B, Q, 1]
        
        offset_input = torch.cat([features_expanded, q_expanded], dim=-1)  # [B, Q, W+1]
        offsets = self.offset_net(offset_input)  # [B, Q, H]
        offsets = offsets.transpose(1, 2)  # [B, H, Q]
        
        # Final prediction: location + scale * offset
        # Offset is scaled by quantile distance from median
        q_centered = (q - 0.5).unsqueeze(1)  # [B, 1, Q]
        
        # Combine location, scale, and offset
        # Use q_centered to ensure symmetric behavior around median
        predictions = median.unsqueeze(-1) + scale.unsqueeze(-1) * offsets * torch.abs(q_centered)
        
        return predictions


class HierarchicalNBEATSAQ(nn.Module):
    """
    N-BEATS based hierarchical quantile forecaster.
    Wraps N-BEATS backbone with hierarchical quantile prediction.
    """
    def __init__(self, num_blocks, num_layers, layer_width, share, size_in, size_out):
        super().__init__()
        from .nbeats import NBEATS, NbeatsBlock
        
        # Create N-BEATS backbone
        self.backbone = NBEATS(
            num_blocks=num_blocks,
            num_layers=num_layers,
            layer_width=layer_width,
            share=share,
            size_in=size_in,
            size_out=layer_width,  # Output features instead of predictions
            block_class=NbeatsBlock
        )
        
        # Wrap with hierarchical predictor
        self.hierarchical = HierarchicalQuantilePredictor(
            backbone=self.backbone,
            size_in=size_in,
            size_out=size_out,
            layer_width=layer_width
        )
    
    def forward(self, x, q):
        """
        Args:
            x: [B, T] - input history
            q: [B, Q] - quantile levels
        
        Returns:
            [B, H, Q] - hierarchical quantile predictions
        """
        return self.hierarchical(x, q)