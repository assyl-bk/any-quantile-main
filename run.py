import argparse
import yaml
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, DictConfig, ListConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.checkpointing import get_checkpoint_path
from utils.model_factory import instantiate


# NumPy 2.0 compatibility: restore removed aliases
if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_


def run(cfg_yaml):
    
    # If config has dotted keys, create from dotlist to get nested structure
    if any('.' in str(k) for k in cfg_yaml.keys()):
        # Config has flattened keys like 'logging.path'
        # We need to restructure it into nested dicts
        nested = {}
        for key, value in cfg_yaml.items():
            parts = key.split('.')
            current = nested
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        cfg = OmegaConf.create(nested)
    else:
        # Config already has proper nested structure
        cfg = OmegaConf.create(cfg_yaml)
    
    print(f"Config type: {type(cfg)}")
    print(f"Config keys: {list(cfg.keys()) if hasattr(cfg, 'keys') else 'No keys method'}")
    print(OmegaConf.to_yaml(cfg))
    
    # Handle logging configuration - use OmegaConf.select for dotted paths
    logging_path = OmegaConf.select(cfg, 'logging.path', default='lightning_logs')
    logging_name = OmegaConf.select(cfg, 'logging.name', default='default')
    
    # Set random seed - handle list of seeds or tuples
    random_seed = OmegaConf.select(cfg, 'random.seed', default=0)
    if isinstance(random_seed, (list, tuple, ListConfig)):
        # If it's a ListConfig or list, take first element
        random_seed = random_seed[0] if len(random_seed) > 0 else 0
    # Ensure it's an integer
    random_seed = int(random_seed)
    
    # Add seed to logger name for unique identification
    logging_name_with_seed = f"{logging_name}-seed{random_seed}"
    
    logger = TensorBoardLogger(save_dir=logging_path, version=logging_name_with_seed, name="")
    
    # Instantiate dataset
    dm = instantiate(cfg.dataset)
    dm.prepare_data()
    
    # Setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=OmegaConf.select(cfg, 'checkpoint.save_top_k', default=2),
        monitor="epoch",
        mode="max",
        filename="model-{epoch}"
    )
    
    pl.seed_everything(random_seed, workers=True)
    
    # Create trainer - convert trainer config to dict and override logger
    # Make sure to resolve=True to handle any interpolations, and None values stay as None
    trainer_config = OmegaConf.select(cfg, 'trainer', default={})
    if trainer_config:
        trainer_kwargs = OmegaConf.to_container(trainer_config, resolve=True)
    else:
        trainer_kwargs = {}
    
    trainer_kwargs['logger'] = logger
    trainer_kwargs['callbacks'] = [lr_monitor, checkpoint_callback]
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Instantiate model
    model = instantiate(OmegaConf.select(cfg, 'model'), cfg=cfg)
    
    # Get checkpoint path for this specific seed
    # Update the logging name in cfg to match the seed-specific logger
    cfg_updated = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.update(cfg_updated, 'logging.name', logging_name_with_seed)
    
    # Get checkpoint path with the updated config
    ckpt_path = get_checkpoint_path(cfg_updated)
    
    # Train
    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    
    # Test
    test_results = trainer.test(datamodule=dm, ckpt_path=OmegaConf.select(cfg, 'checkpoint.ckpt_path'))
    
    return test_results


def main(config_path: str, overrides: list = []):
    
    if not torch.cuda.is_available():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CUDA is NOT available !!!")
        print("!!! CUDA is NOT available !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
    with open(config_path) as f:
        cfg_yaml = yaml.unsafe_load(f)
    
    # Check if we have multiple seeds
    random_seed = cfg_yaml.get('random.seed') or cfg_yaml.get('random', {}).get('seed', 0)
    
    # Handle tuple/list of seeds
    if isinstance(random_seed, (list, tuple)):
        seeds = list(random_seed)
        print(f"\n{'='*80}")
        print(f"Running experiments for {len(seeds)} seeds: {seeds}")
        print(f"{'='*80}\n")
        
        all_test_results = []
        
        for seed_idx, seed in enumerate(seeds):
            print(f"\n{'='*80}")
            print(f"SEED {seed_idx + 1}/{len(seeds)}: {seed}")
            print(f"{'='*80}\n")
            
            # Create a copy of config with single seed
            cfg_copy = cfg_yaml.copy()
            cfg_copy['random.seed'] = seed
            
            # For multi-seed runs, each seed should train from scratch
            # Set resume_ckpt to None to prevent loading from checkpoint
            if 'checkpoint.resume_ckpt' in cfg_copy:
                cfg_copy['checkpoint.resume_ckpt'] = None
            elif 'checkpoint' in cfg_copy and isinstance(cfg_copy['checkpoint'], dict):
                cfg_copy['checkpoint']['resume_ckpt'] = None
            
            # Run experiment for this seed
            test_results = run(cfg_copy)
            all_test_results.append({
                'seed': seed,
                'results': test_results
            })
        
        # Print summary of all seeds
        print(f"\n{'='*80}")
        print(f"SUMMARY OF ALL SEEDS")
        print(f"{'='*80}\n")
        
        # Collect metrics across seeds
        metrics_by_name = {}
        
        for result in all_test_results:
            print(f"\nSeed {result['seed']}:")
            if result['results']:
                for metric_dict in result['results']:
                    for metric_name, metric_value in metric_dict.items():
                        print(f"  {metric_name}: {metric_value}")
                        # Collect for aggregation
                        if metric_name not in metrics_by_name:
                            metrics_by_name[metric_name] = []
                        metrics_by_name[metric_name].append(float(metric_value))
            else:
                print("  No results available")
        
        # Print aggregate statistics
        if metrics_by_name:
            print(f"\n{'='*80}")
            print(f"AGGREGATE STATISTICS (Mean ± Std)")
            print(f"{'='*80}\n")
            for metric_name, values in sorted(metrics_by_name.items()):
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{metric_name:30s}: {mean_val:12.6f} ± {std_val:10.6f}")
        
        print(f"\n{'='*80}\n")
    else:
        # Single seed
        run(cfg_yaml)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False, description="Experiment")
    parser.add_argument('--config', type=str, 
                        help='Path to the experiment configuration file', 
                        default='config/config.yaml')
    parser.add_argument("overrides", nargs="*",
                        help="Any key=value arguments to override config values (use dots for.nested=overrides)")
    args = parser.parse_args()

    main(config_path=args.config, overrides=args.overrides)