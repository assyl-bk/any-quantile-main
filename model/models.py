import pandas as pd
import numpy as np
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torch

# from hydra.utils import instantiate
from utils.model_factory import instantiate

from torchmetrics import MeanSquaredError, MeanAbsoluteError
from metrics import SMAPE, MAPE, CRPS, Coverage

from modules import MLP
from losses import MonotonicityLoss
    
    
class MlpForecaster(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.backbone = instantiate(cfg.model.nn.backbone)
        self.init_metrics()
        self.loss = instantiate(cfg.model.loss)
        
    def init_metrics(self):
        self.train_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        self.train_smape = SMAPE()
        self.val_smape = SMAPE()
        self.test_smape = SMAPE()
        self.val_mape = MAPE()
        self.test_mape = MAPE()
        
    def shared_forward(self, x):
        history = x['history'][:, -self.cfg.model.input_horizon_len:]
        forecast = self.backbone(history)   
        return {'forecast': forecast}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['forecast']

    def training_step(self, batch, batch_idx):
        net_output = self.shared_forward(batch)
        y_hat = net_output['forecast']
        loss = self.loss(y_hat, batch['target']) 
        
        batch_size=batch['history'].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.train_mse(y_hat, batch['target'])
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.train_mae(y_hat, batch['target'])
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        net_output = self.shared_forward(batch)
        y_hat = net_output['forecast']
        
        self.val_mse(y_hat, batch['target'])
        self.val_mae(y_hat, batch['target'])
        self.val_smape(y_hat, batch['target'])
                
        batch_size=batch['history'].shape[0]
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/smape", self.val_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
    def test_step(self, batch, batch_idx):
        net_output = self.shared_forward(batch)
        y_hat = net_output['forecast']
        
        self.test_mse(y_hat, batch['target'])
        self.test_mae(y_hat, batch['target'])
        self.test_smape(y_hat, batch['target'])
        self.test_mape(y_hat, batch['target'])
                
        batch_size=batch['history'].shape[0]
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/smape", self.test_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/mape", self.test_mape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer, self.parameters())
        scheduler = instantiate(self.cfg.model.scheduler, optimizer)
        if scheduler is not None:
            optimizer = {"optimizer": optimizer, 
                         "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer
    
    
class AnyQuantileForecaster(MlpForecaster):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.train_crps = CRPS()
        self.val_crps = CRPS()
        self.test_crps = CRPS()

        self.train_coverage = Coverage(level=0.95)
        self.val_coverage = Coverage(level=0.95)
        self.test_coverage = Coverage(level=0.95)
        
    def shared_forward(self, x):
        history = x['history'][:, -self.cfg.model.input_horizon_len:]
        q = x['quantiles']

        x_max = torch.abs(history).max(dim=-1, keepdims=True)[0]
        if self.cfg.model.max_norm:
            x_max[x_max == 0] = 1
        else:
            x_max[x_max >= 0] = 1
        history = history / x_max
        
        forecast = self.backbone(history, q)
        return {'forecast': forecast * x_max[..., None], 'quantiles': q}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['forecast']

    def training_step(self, batch, batch_idx):
        batch_size = batch['history'].shape[0]
        if self.cfg.model.q_sampling == 'fixed_in_batch':
            q = torch.rand(1)
            batch['quantiles'] = (q * torch.ones(batch_size, 1)).to(batch['history'])
        elif self.cfg.model.q_sampling == 'random_in_batch':
            if self.cfg.model.q_distribution == 'uniform':
                batch['quantiles'] = torch.rand(batch_size, 1).to(batch['history'])
            elif self.cfg.model.q_distribution == 'beta':
                batch['quantiles'] = torch.Tensor(np.random.beta(self.cfg.model.q_parameter, self.cfg.model.q_parameter, 
                                                                 size=(batch_size, 1))).to(batch['history'])
            else:
                assert False, f"Option {self.cfg.model.q_distribution} is not implemented"
        else:
            assert False, f"Option {self.cfg.model.q_sampling} is not implemented"
        
        net_output = self.shared_forward(batch)
        y_hat = net_output['forecast']
        quantiles = net_output['quantiles'][:,None]
        center_idx = y_hat.shape[-1]
        assert center_idx % 2 == 1, "Number of quantiles must be odd"
        center_idx = center_idx // 2
        
        loss = self.loss(y_hat, batch['target'], q=quantiles) 
        
        batch_size=batch['history'].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.train_mse(y_hat[..., center_idx], batch['target'])
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.train_mae(y_hat[..., center_idx], batch['target'])
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.train_crps(y_hat, batch['target'], q=quantiles)
        self.log("train/crps", self.train_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch['quantiles'] = self.val_coverage.add_evaluation_quantiles(batch['quantiles'])
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast']
        quantiles = net_output['quantiles'][:,None]
        
        self.val_mse(y_hat[..., 0].contiguous(), batch['target'])
        self.val_mae(y_hat[..., 0].contiguous(), batch['target'])
        self.val_smape(y_hat[..., 0].contiguous(), batch['target'])
        self.val_mape(y_hat[..., 0].contiguous(), batch['target'])
        self.val_crps(y_hat, batch['target'], q=quantiles)
        self.val_coverage(y_hat, batch['target'], q=quantiles)
                
        batch_size=batch['history'].shape[0]
        self.log("val/mse", self.val_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/smape", self.val_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/mape", self.val_mape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("val/crps", self.val_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f"val/coverage-{self.val_coverage.level}", self.val_coverage, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
    def test_step(self, batch, batch_idx):
        batch['quantiles'] = self.test_coverage.add_evaluation_quantiles(batch['quantiles'])
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast']
        quantiles = net_output['quantiles'][:,None]
        
        self.test_mse(y_hat[..., 0].contiguous(), batch['target'])
        self.test_mae(y_hat[..., 0].contiguous(), batch['target'])
        self.test_smape(y_hat[..., 0].contiguous(), batch['target'])
        self.test_mape(y_hat[..., 0].contiguous(), batch['target'])
        self.test_crps(y_hat, batch['target'], q=quantiles)
        self.test_coverage(y_hat, batch['target'], q=quantiles)
                
        batch_size=batch['history'].shape[0]
        self.log("test/mse", self.test_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/smape", self.test_smape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/mape", self.test_mape, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log("test/crps", self.test_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        self.log(f"test/coverage-{self.test_coverage.level}", self.test_coverage, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)


class AnyQuantileForecasterWithMonotonicity(AnyQuantileForecaster):
    """Any-Quantile Forecaster with Monotonicity Constraint"""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_monotonicity_loss = cfg.model.get('use_monotonicity_loss', False)
        self.monotone_weight = cfg.model.get('monotone_weight', 0.1)
        self.monotone_margin = cfg.model.get('monotone_margin', 0.0)
        
    def monotonicity_loss(self, y_hat, quantiles):
        """
        Enforce that predictions for higher quantiles are >= predictions for lower quantiles
        y_hat: BxHxQ
        quantiles: Bx1xQ
        """
        # Sort by quantile values
        sorted_indices = torch.argsort(quantiles, dim=-1)
        sorted_y_hat = torch.gather(y_hat, -1, sorted_indices.expand_as(y_hat))
        
        # Compute differences between consecutive quantiles
        diffs = sorted_y_hat[..., 1:] - sorted_y_hat[..., :-1]
        
        # Penalize negative differences (violations)
        violations = F.relu(self.monotone_margin - diffs)
        return violations.mean()
    
    def training_step(self, batch, batch_idx):
        # Generate random quantiles
        batch_size = batch['history'].shape[0]
        num_quantiles = self.cfg.model.get('num_train_quantiles', 1)
        
        if self.cfg.model.q_sampling == 'random_in_batch':
            if self.cfg.model.q_distribution == 'uniform':
                batch['quantiles'] = torch.rand(batch_size, num_quantiles).to(batch['history'])
            elif self.cfg.model.q_distribution == 'beta':
                batch['quantiles'] = torch.Tensor(np.random.beta(
                    self.cfg.model.q_parameter, 
                    self.cfg.model.q_parameter, 
                    size=(batch_size, num_quantiles)
                )).to(batch['history'])
            else:
                assert False, f"Option {self.cfg.model.q_distribution} not implemented"
        else:
            assert False, f"Option {self.cfg.model.q_sampling} not implemented"
        
        net_output = self.shared_forward(batch)
        
        y_hat = net_output['forecast']  # BxHxQ
        quantiles = net_output['quantiles'][:, None]  # Bx1xQ
        
        # Main quantile loss
        main_loss = self.loss(y_hat, batch['target'], q=quantiles)
        
        # Add monotonicity loss if enabled
        if self.use_monotonicity_loss and num_quantiles > 1:
            mono_loss = self.monotonicity_loss(y_hat, quantiles)
            loss = main_loss + self.monotone_weight * mono_loss
            
            self.log("train/mono_loss", mono_loss, on_step=False, on_epoch=True, 
                     prog_bar=False, logger=True, batch_size=batch_size)
        else:
            loss = main_loss
        
        batch_size = batch['history'].shape[0]
        self.log("train/loss", loss, on_step=True, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        # Metrics on median quantile
        median_idx = num_quantiles // 2
        self.train_mse(y_hat[..., median_idx], batch['target'])
        self.log("train/mse", self.train_mse, on_step=False, on_epoch=True, 
                 prog_bar=True, logger=True, batch_size=batch_size)
        
        self.train_mae(y_hat[..., median_idx], batch['target'])
        self.log("train/mae", self.train_mae, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        self.train_crps(y_hat, batch['target'], q=quantiles)
        self.log("train/crps", self.train_crps, on_step=False, on_epoch=True, 
                 prog_bar=False, logger=True, batch_size=batch_size)
        
        return loss
    
class GeneralForecaster(MlpForecaster):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.time_series_projection_in = torch.nn.Linear(1, cfg.model.nn.backbone.d_model)
        self.time_series_projection_out = torch.nn.Linear(cfg.model.nn.backbone.d_model, 1)
        self.shortcut = torch.nn.Linear(cfg.model.nn.backbone.size_in, cfg.model.nn.backbone.size_out)
        self.time_embedding = torch.nn.Embedding(2000, cfg.model.nn.embedding_dim)
        self.time_series_id = torch.nn.Embedding(cfg.model.nn.time_series_id_num, cfg.model.nn.embedding_dim)
        
    def shared_forward(self, x):
        history = x['history'][:, -self.cfg.model.input_horizon_len:]
        
        t_h = torch.arange(self.cfg.model.input_horizon_len, dtype=torch.int64)[None].to(history.device)
        t_t = torch.arange(x['time_features_target'].shape[1], dtype=torch.int64)[None].to(history.device) + self.cfg.model.input_horizon_len
        
        time_features_tgt = torch.repeat_interleave(self.time_embedding(t_t), repeats=history.shape[0], dim=0)
        time_features_src = self.time_embedding(t_h)
        
        xf_input = time_features_tgt
        xt_input = time_features_src + self.time_series_projection_in(history.unsqueeze(-1))
        xs_input = 0.0 * self.time_series_id(x['series_id'])
        
        backbone_output = self.backbone(xt_input=xt_input, xf_input=xf_input, xs_input=xs_input)   
        backbone_output = self.time_series_projection_out(backbone_output)
        forecast = backbone_output[..., 0] + history.mean(dim=-1, keepdims=True) + self.shortcut(history)
        return {'forecast': forecast}

    def forward(self, x):
        out = self.shared_forward(x)
        return out['forecast']