from torchmetrics import Metric
from losses import MQLoss
import torch


def _divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Auxiliary function to handle divide by 0
    """
    # Add epsilon to denominator to prevent division by zero
    b_safe = torch.where(torch.abs(b) < 1e-8, torch.ones_like(b), b)
    div = a / b_safe
    # Replace any remaining inf/nan with 0
    div = torch.nan_to_num(div, nan=0.0, posinf=0.0, neginf=0.0)
    return div


class Coverage(Metric):
    def __init__(self, dist_sync_on_step=False, level=0.95):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.level_low = (1.0-level)/2
        self.level_high = 1.0 - self.level_low
        self.level = level

    def add_evaluation_quantiles(self, quantiles: torch.Tensor):
        quantiles_metric = torch.Tensor([self.level_high, self.level_low])
        quantiles_metric = torch.repeat_interleave(quantiles_metric[None], repeats=quantiles.shape[0], dim=0)
        quantiles_metric = quantiles_metric.to(quantiles)
        return torch.cat([quantiles, quantiles_metric], dim=-1)

    def update(self, preds: torch.Tensor, target: torch.Tensor, q: torch.Tensor) -> None:
        if target.dim() != preds.dim():
            target = target[..., None]

        # Find quantiles matching the target levels
        mask_high = torch.isclose(q, torch.tensor(self.level_high).to(q.device), atol=1e-6)
        mask_low = torch.isclose(q, torch.tensor(self.level_low).to(q.device), atol=1e-6)
        
        num_high = mask_high.sum(dim=-1, keepdims=True).clamp(min=1)  # Prevent division by zero
        num_low = mask_low.sum(dim=-1, keepdims=True).clamp(min=1)
        
        preds_high = (preds * mask_high).sum(dim=-1, keepdims=True) / num_high
        preds_low = (preds * mask_low).sum(dim=-1, keepdims=True) / num_low
                
        self.numerator += ((target < preds_high) & (target >= preds_low)).sum()
        self.denominator += torch.numel(target)

    def compute(self):
        if self.denominator == 0:
            return torch.tensor(0.0)
        return self.numerator / self.denominator
        

class CRPS(Metric):
    def __init__(self, dist_sync_on_step=False, horizon=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.horizon = horizon

    def update(self, preds: torch.Tensor, target: torch.Tensor, q: torch.Tensor) -> None:
        """ Compute multi-quantile loss function
        
        :param preds: BxHxQ tensor of predicted values, Q is the number of quantiles
        :param target: BxHx1, or BxH tensor of target values
        :param q: BxHxQ or Bx1xQ tensor of quantiles telling which quantiles input predictions correspond to
        :return: value of multi-quantile loss function
        """
        
        if target.dim() != preds.dim():
            target = target[..., None]
        
        # Ensure q has the same shape as preds
        if q.dim() == 2:  # Bx1xQ -> BxHxQ
            q = q.unsqueeze(1).expand(-1, preds.shape[1], -1)
            
        # Compute pinball loss
        errors = target - preds
        pinball = torch.where(errors >= 0, q * errors, (q - 1) * errors)
        
        if self.horizon is None:
            loss_value = pinball.mean()
            num_elements = torch.numel(preds)
        else:
            loss_value = pinball[:, self.horizon].mean()
            num_elements = torch.numel(preds[:, self.horizon])
        
        # Check for NaN before accumulating
        if not torch.isnan(loss_value) and not torch.isinf(loss_value):
            self.numerator += loss_value * num_elements
            self.denominator += num_elements

    def compute(self):
        if self.denominator == 0:
            return torch.tensor(float('nan'))
        return 2 * (self.numerator / self.denominator)

    
class MAPE(Metric):
    def __init__(self, dist_sync_on_step=False, horizon=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("nsamples", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.horizon = horizon

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        
        if self.horizon is None:
            # Only compute MAPE where target is not too small (avoid division by near-zero)
            mask = torch.abs(target) > 1e-3
            if mask.sum() > 0:
                mape = _divide_no_nan(torch.abs(target[mask] - preds[mask]), torch.abs(target[mask]))
                self.total_error += mape.sum()
                self.nsamples += mask.sum()
        else:
            target_h = target[:, self.horizon]
            preds_h = preds[:, self.horizon]
            mask = torch.abs(target_h) > 1e-3
            if mask.sum() > 0:
                mape = _divide_no_nan(torch.abs(target_h[mask] - preds_h[mask]), torch.abs(target_h[mask]))
                self.total_error += mape.sum()
                self.nsamples += mask.sum()

    def compute(self):
        if self.nsamples == 0:
            return torch.tensor(float('nan'))
        return 100 * (self.total_error / self.nsamples)
    

class SMAPE(Metric):
    def __init__(self, dist_sync_on_step=False, horizon=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("smape", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("nsamples", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.horizon = horizon

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        if self.horizon is None:
            smape = 2 * _divide_no_nan(torch.abs(target - preds), torch.abs(target) + torch.abs(preds))
            self.smape += smape.sum()
            self.nsamples += torch.numel(smape)
        else:
            smape = 2 * _divide_no_nan(torch.abs(target[:, self.horizon] - preds[:, self.horizon]), 
                                       torch.abs(target[:, self.horizon]) + torch.abs(preds[:, self.horizon]))
            self.smape += smape.sum()
            self.nsamples += torch.numel(target[:, self.horizon])

    def compute(self):
        if self.nsamples == 0:
            return torch.tensor(float('nan'))
        return 100 * (self.smape / self.nsamples)

    
class WAPE(Metric):
    def __init__(self, dist_sync_on_step=False, horizon=None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("numerator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("denominator", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.horizon = horizon

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        if self.horizon is None:
            self.numerator += torch.abs(target - preds).sum()
            self.denominator += torch.abs(target).sum()
        else:
            self.numerator += torch.abs(target[:, self.horizon] - preds[:, self.horizon]).sum()
            self.denominator += torch.abs(target[:, self.horizon]).sum()

    def compute(self):
        if self.denominator == 0:
            return torch.tensor(float('nan'))
        return 100 * (self.numerator / self.denominator)