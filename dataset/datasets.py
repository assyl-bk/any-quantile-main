from typing import List
from torch.utils.data import Dataset, DataLoader, Subset
from glob import glob
import os
import pathlib
import pandas as pd
import numpy as np
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torch

import bisect

import subprocess
import shutil
from glob import glob
import logging
import zipfile

import contextlib

from datasetsforecast.long_horizon import LongHorizon

logger = logging.getLogger("dataset")


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collate_fn_flat_deal(batch):
    out = {}
    for b in batch:
        for k, bv in b.items():
            v = out.get(k, [])
            v.append(bv)
            out[k] = v
            
    for k,v in out.items():
        v = np.concatenate(v)
        if type(v[0]) not in [np.str_, pd.Timestamp]:
            v = torch.as_tensor(v)
        out[k] = v
    return out


class ElectricityUnivariateDataset(Dataset):
    def __init__(self, 
                 name: str, 
                 split: str, 
                 split_start: pd.Timestamp = None,
                 split_end: pd.Timestamp = None,
                 horizon_length: int = 720, 
                 history_length: int = 720,
                 fillna: str = 'ffill',
                 step: int = 1,
                 init_seed: int = 0):
        super().__init__()
        self.name = name
        self.split = split
        self.split_end = split_end
        self.split_start = split_start
        self.horizon_length = horizon_length
        self.history_length = history_length
        self.fillna = fillna
        self.step = step
        
        path = os.path.join('./data/electricity/datasets', name.lower(), 'M/df_y.csv')
        df = pd.read_csv(path)
        df.ds = pd.to_datetime(df.ds)
        df = df.pivot(index='ds', columns='unique_id', values='y')
        cols = {int(c.split('Var')[-1]): c for c in df.columns}
        df = df[[cols[k+1] for k in range(len(cols))]]
        
        if isinstance(split_end, pd.Timestamp) or isinstance(split_end, str):
            if df.index.max() < pd.to_datetime(self.split_start):
                raise Exception(f'Start date must fall before dataset end')
            start = np.argmax(df.index >= self.split_start)
            start = max(start - self.history_length, 0)
            end = np.sum(df.index < self.split_end)
        elif isinstance(split_end, float):
            if split_end <= 0 or split_end > 1.0:
                raise Exception(f'Float split_end must be in (0, 1]. Unacceptable value {split_end} found')
            if split_start >= split_end:
                raise Exception(f'split_end must be greater than split_start. \
                                  Unacceptable values {split_start} >= {split_end} found')
            start = max(int(split_start * len(df)) - self.history_length, 0)
            end = int(split_end * len(df))
        else:
            raise Exception(f'Unimplemented split end option {split_end.__class__}')
        
        df = df[start:end]
        self.df = df.astype(np.float32)
        
        self.tot_window_len = horizon_length + history_length
        
        if self.fillna is not None:
            if isinstance(self.fillna, str):
                self.df.fillna(method=self.fillna, inplace=True)
            elif np.isreal(self.fillna):
                self.df.fillna(value=self.fillna, inplace=True)
            else:
                raise NotImplementedError(f"Fill NaN method {self.fillna} is not implemented")

        # Compute the number of windows available in each column
        self.num_windows = (~self.df.isnull()).sum(axis=0) - self.tot_window_len + 1
        # Drop columns with less than 1 window
        self.df.drop(self.df.columns[self.num_windows <= 0], axis=1, inplace=True)
        # Recompute the number of windows available in each column after skipping empty columns
        num_samples = (~self.df.isnull()).sum(axis=0)
        self.num_windows = (num_samples - self.tot_window_len) // self.step + 1
        # Compute column boundaries in terms of available windows in each column
        self.cum_num_windows = self.num_windows.cumsum().values.ravel()
        self.num_nulls = self.df.isnull().sum(axis=0).values.ravel()
        
        self.series_timestamps = pd.Series(self.df.index)
        self.series_dow = self.series_timestamps.dt.dayofweek.astype(np.int64)
        self.series_month = self.series_timestamps.dt.month.astype(np.int64)
        self.series_day = self.series_timestamps.dt.day.astype(np.int64)
        
        self.time_features = np.stack([self.series_day, 
                                       int(31) + self.series_dow, # shift by 31 days
                                       int(31) + int(7) + self.series_month]  # shift by 31 days and 7 days of week
                                     ).T
        
        self.init_seed = init_seed
        self.test_quantiles = np.array([50, 0.1, 1, 5, 10, 25, 75, 90, 95, 99, 99.9], dtype=np.float32)/100        
 
    def __getitem__(self, index):
        column_idx = bisect.bisect_right(self.cum_num_windows, index)
        window_idx = self.num_nulls[column_idx] + self.step * index - (self.step * self.cum_num_windows[column_idx-1] if column_idx > 0 else 0)
        ts = self.df.values[window_idx:window_idx+self.tot_window_len, column_idx]

        time_features = self.time_features[window_idx:window_idx+self.tot_window_len]
        
        # This is needed for quantile randomization to reduce variance, 
        # but also make sure that samples are synchronized for ensemble calculations
        with temp_seed(self.init_seed + index):
            random_quantiles = np.random.rand(100).astype(np.float32)
        test_quantiles = np.concatenate([self.test_quantiles, random_quantiles]).astype(np.float32)
        
        item = {
            'target': ts[self.history_length:][None],
            'history': ts[:self.history_length][None],
            'series_id': np.array([column_idx], dtype=np.int64),
            'time_features_target': time_features[self.history_length:][None],
            'time_features_history': time_features[:self.history_length][None],
            'quantiles': test_quantiles[None],
        }
        
        return item
    
    def __len__(self):
        return self.cum_num_windows[-1]
    
    
class ElectricityUnivariateDataModule(pl.LightningDataModule):
    def __init__(self, 
                 name: str = 'MHLV', 
                 train_batch_size: int = 128, 
                 eval_batch_size: int = None,
                 num_workers: int = 4,
                 persistent_workers: bool = True,
                 horizon_length: int = 720,
                 history_length: int = 720,
                 split_proportions: List[float] = None,
                 split_boundaries: List[str] = None,
                 fillna: str = None,
                 train_step: int = 1,
                 eval_step: int = 1,
                ):
        super().__init__()
        self.name = name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = train_batch_size
        if eval_batch_size is not None:
            self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.train_step = train_step
        self.eval_step = eval_step
        
        if (split_proportions is not None) and (split_boundaries is not None):
            raise Exception(f'Only split_proportions or split_boundaries can be set, not both')   
        if split_proportions is not None:
            self.split_boundaries = np.array([0] + split_proportions).cumsum()
            assert self.split_boundaries[-1] == 1, "Split proportions must sum up to 1"
        else:
            self.split_boundaries = split_boundaries
            
        self.fillna = fillna
    
    @staticmethod
    def prepare_data():
        logger.info("Downloading datasets")
        datasets_path = "data/electricity/datasets/"
        pathlib.Path(datasets_path).mkdir(parents=True, exist_ok=True)
        # If datasets are already present, skip download
        datasets = glob(os.path.join(datasets_path, "*.zip"))
        if len(datasets) == 0:
            # Try to use gsutil if available
            gsutil_path = shutil.which("gsutil")
            if gsutil_path is None:
                msg = (
                    "gsutil is not installed or not on PATH. The script attempted to download datasets to "
                    f"'{datasets_path}' but found no .zip files.\n"
                    "Options:\n"
                    "  1) Install the Google Cloud SDK (gsutil) and re-run the script. On Debian/Ubuntu: 'sudo apt-get install -y google-cloud-sdk' or follow https://cloud.google.com/sdk/docs/install.\n"
                    "  2) Manually download the datasets archive and place the .zip files under 'data/electricity/datasets/'.\n"
                    "  3) If you have gsutil but it's not on PATH, ensure it's available to the environment running Python.\n"
                )
                logger.error(msg)
                raise RuntimeError(msg)

            try:
                proc = subprocess.run([gsutil_path, "-m", "rsync", "gs://electricity-datasets", datasets_path],
                                      capture_output=True, check=True)
                # gsutil writes progress to stderr
                logger.info(proc.stderr.decode(errors='ignore'))
            except subprocess.CalledProcessError as e:
                logger.error("gsutil failed to sync datasets: %s", e)
                logger.error("stdout: %s", e.stdout.decode(errors='ignore') if e.stdout else "")
                logger.error("stderr: %s", e.stderr.decode(errors='ignore') if e.stderr else "")
                raise
            datasets = glob(os.path.join(datasets_path, "*.zip"))

        if len(datasets) > 0:
            logger.info("Unzipping datasets")
            for d in datasets:
                try:
                    # Use Python's zipfile module for cross-platform compatibility
                    with zipfile.ZipFile(d, 'r') as zip_ref:
                        # Extract to the same directory as the zip file
                        extract_path = os.path.dirname(d)
                        zip_ref.extractall(extract_path)
                        logger.info(f"Extracted {d} to {extract_path}")
                except zipfile.BadZipFile as e:
                    logger.error("Failed to unzip %s: %s (not a valid zip file)", d, e)
                    raise
                except Exception as e:
                    logger.error("Failed to unzip %s: %s", d, e)
                    raise

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = ElectricityUnivariateDataset(name=self.name, split='train', 
                                                              split_start=self.split_boundaries[0],
                                                              split_end=self.split_boundaries[1],
                                                              horizon_length=self.horizon_length,
                                                              history_length = self.history_length,
                                                              fillna=self.fillna,
                                                              step = self.train_step)
            self.val_dataset = ElectricityUnivariateDataset(name=self.name, split='val', 
                                                            split_start=self.split_boundaries[1],
                                                            split_end=self.split_boundaries[2],
                                                            horizon_length=self.horizon_length,
                                                            history_length = self.history_length,
                                                            fillna=self.fillna,
                                                            step = self.eval_step)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = ElectricityUnivariateDataset(name=self.name, split='test', 
                                                             split_start=self.split_boundaries[2],
                                                             split_end=self.split_boundaries[3],
                                                             horizon_length=self.horizon_length,
                                                             history_length = self.history_length,
                                                             fillna=np.Inf,
                                                             step = self.eval_step)
        if stage == "predict":
            self.predict_dataset = ElectricityUnivariateDataset(name=self.name, split='test', 
                                                                split_start=self.split_boundaries[2],
                                                                split_end=self.split_boundaries[3],
                                                                horizon_length=self.horizon_length,
                                                                history_length = self.history_length,
                                                                fillna=np.Inf,
                                                                step = self.eval_step)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, 
                          shuffle=True, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, 
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size,
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.eval_batch_size, 
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)


def get_zip(url: str, name: str, path: str="/tmp/cache", unpack: bool=True):
    dir = os.path.join(path, name)
    os.makedirs(dir, exist_ok=True)
    if url.startswith("kaggle"):
        os.system(f"{url} --path {dir}")
        zip_fname = os.path.join(dir, os.path.basename(url) + ".zip")
    elif url.startswith("https://"):
        zip_fname = f"{os.path.join(dir, os.path.basename(url))}"
        os.system(f"wget -c --read-timeout=5 --tries=0 {url} -O {zip_fname}")
    else:
        assert False, f"URL download method is not implemented for {url}"
    if unpack:
        if zip_fname.endswith('.zip'):
            os.system(f"unzip -u {zip_fname} -d {dir}")
        elif zip_fname.endswith('.tgz') or zip_fname.endswith('.tar.gz'):
            os.system(f"tar -xvzf {zip_fname} -C {dir}")
    return dir


class EMHIRESUnivariateDataModule(pl.LightningDataModule):
    def __init__(self, 
                 name: str = 'MHLV', 
                 train_batch_size: int = 128, 
                 eval_batch_size: int = None,
                 num_workers: int = 4,
                 persistent_workers: bool = True,
                 horizon_length: int = 720,
                 history_length: int = 720,
                 split_proportions: List[float] = None,
                 split_boundaries: List[str] = None,
                 fillna: str = None,
                 train_step: int = 1,
                 eval_step: int = 1,
                ):
        super().__init__()
        self.name = name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = train_batch_size
        if eval_batch_size is not None:
            self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.train_step = train_step
        self.eval_step = eval_step
        
        if (split_proportions is not None) and (split_boundaries is not None):
            raise Exception(f'Only split_proportions or split_boundaries can be set, not both')   
        if split_proportions is not None:
            self.split_boundaries = np.array([0] + split_proportions).cumsum()
            assert self.split_boundaries[-1] == 1, "Split proportions must sum up to 1"
        else:
            self.split_boundaries = split_boundaries
            
        self.fillna = fillna
    
    @staticmethod
    def prepare_data():
        logger.info("Downloading datasets")
        datasets_path = "data/emhires/datasets/"
        pathlib.Path(datasets_path).mkdir(parents=True, exist_ok=True)

        logger.info("Download EMHIRES PV")
        emhires_pv_parquet = 'data/emhires/pv_n2.parquet'
        if os.path.isfile(emhires_pv_parquet):
            logger.info(f"EMHIRES PV cached version found in {emhires_pv_parquet}")
        else:
            get_zip(url='https://zenodo.org/records/8340501/files/EMHIRES_PV_NUTS2.zip',
                    name='emhires_pv', path="data/downloads", unpack=True)
            logger.info("Clean, format, save EMHIRES PV")
            df = pd.read_excel('data/downloads/emhires_pv/EMHIRES_PVGIS_TSh_CF_n2_19862015_reformatt.xlsx')
            df.drop('SE33', axis=1, inplace=True)
            df['time_step'] = pd.date_range(start='1986-01-01', periods=len(df), freq='h')
            df = df.rename({'time_step': 'ds'}, axis=1).set_index('ds')
            df.to_parquet(emhires_pv_parquet)

        logger.info("Download EMHIRES Wind")
        emhires_wind_parquet = 'data/emhires/wind_n2.parquet'
        if os.path.isfile(emhires_wind_parquet):
            logger.info(f"EMHIRES WIND cached version found in {emhires_wind_parquet}")
        else:
            get_zip(url='https://zenodo.org/records/8340501/files/EMHIRES_WIND_ONSHORE_NUTS2.zip',
                    name='emhires_wind', path="data/downloads", unpack=True)
            logger.info("Clean, format, save EMHIRES WIND")
            df = pd.read_excel('data/downloads/emhires_wind/EMHIRES_WIND_NUTS2_June2019.xlsx')
            df.drop(['LI00', 'MT00'], axis=1, inplace=True)
            df['time_step'] = pd.date_range(start='1986-01-01', periods=len(df), freq='h')
            df = df.rename({'time_step': 'ds'}, axis=1).set_index('ds')
            df.to_parquet(emhires_wind_parquet)

        assert False

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = ElectricityUnivariateDataset(name=self.name, split='train', 
                                                              split_start=self.split_boundaries[0],
                                                              split_end=self.split_boundaries[1],
                                                              horizon_length=self.horizon_length,
                                                              history_length = self.history_length,
                                                              fillna=self.fillna,
                                                              step = self.train_step)
            self.val_dataset = ElectricityUnivariateDataset(name=self.name, split='val', 
                                                            split_start=self.split_boundaries[1],
                                                            split_end=self.split_boundaries[2],
                                                            horizon_length=self.horizon_length,
                                                            history_length = self.history_length,
                                                            fillna=self.fillna,
                                                            step = self.eval_step)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = ElectricityUnivariateDataset(name=self.name, split='test', 
                                                             split_start=self.split_boundaries[2],
                                                             split_end=self.split_boundaries[3],
                                                             horizon_length=self.horizon_length,
                                                             history_length = self.history_length,
                                                             fillna=np.Inf,
                                                             step = self.eval_step)
        if stage == "predict":
            self.predict_dataset = ElectricityUnivariateDataset(name=self.name, split='test', 
                                                                split_start=self.split_boundaries[2],
                                                                split_end=self.split_boundaries[3],
                                                                horizon_length=self.horizon_length,
                                                                history_length = self.history_length,
                                                                fillna=np.Inf,
                                                                step = self.eval_step)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, 
                          shuffle=True, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, 
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size,
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.eval_batch_size, 
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)

    
class LongHorizonUnivariateDataset(Dataset):
    def __init__(self, 
                 name: str, 
                 split: str, 
                 split_start: pd.Timestamp = None,
                 split_end: pd.Timestamp = None,
                 horizon_length: int = 720, 
                 history_length: int = 720):
        super().__init__()
        self.name = name
        self.split = split
        self.split_end = split_end
        self.split_start = split_start
        self.horizon_length = horizon_length
        self.history_length = history_length
        
        df, _, _ = LongHorizon.load(directory='./data', group=name)
        df.ds = pd.to_datetime(df.ds)
        df = df.pivot(index='ds', columns='unique_id', values='y')
        
        if isinstance(split_end, pd.Timestamp) or isinstance(split_end, str):
            flag = (df.index <= self.split_end) & (df.index > self.split_start)
            df = df[flag]
        elif isinstance(split_end, float):
            if split_end <= 0 or split_end > 1.0:
                raise Exception(f'Float split_end must be in (0, 1]. Unacceptable value {split_end} found')
            if split_start >= split_end:
                raise Exception(f'split_end must be greater than split_start. \
                                  Unacceptable values {split_start} >= {split_end} found')
            start = max(int(split_start * len(df)) - self.history_length, 0)
            end = int(split_end * len(df))
            df = df[start:end]
        else:
            raise Exception(f'Unimplemented split end option {split_end.__class__}')
            
        self.df = df.astype(np.float32)
        
        self.tot_window_len = horizon_length + history_length
        self.num_windows = len(df) - self.tot_window_len + 1
        
        self.series_timestamps = pd.Series(self.df.index)
        self.series_dow = self.series_timestamps.dt.dayofweek.astype(np.int64)
        self.series_month = self.series_timestamps.dt.month.astype(np.int64)
        self.series_day = self.series_timestamps.dt.day.astype(np.int64)
        
        self.time_features = np.stack([self.series_day, 
                                       int(31) + self.series_dow, # shift by 31 days
                                       int(31) + int(7) + self.series_month]  # shift by 31 days and 7 days of week
                                     ).T
        
    def __getitem__(self, index):
        column_idx = index // self.num_windows
        window_idx = index - column_idx*self.num_windows
        ts = self.df.values[window_idx:window_idx+self.tot_window_len, column_idx]
                
        time_features = self.time_features[window_idx:window_idx+self.tot_window_len]
        
        item = {
            'target': ts[self.history_length:][None],
            'history': ts[:self.history_length][None],
            'series_id': np.array([column_idx], dtype=np.int64),
            'time_features_target': time_features[self.history_length:][None],
            'time_features_history': time_features[:self.history_length][None],
        }
        
        return item
    
    def __len__(self):
        return self.num_windows * len(self.df.columns)
    
    
class LongHorizonUnivariateDataModule(pl.LightningDataModule):
    def __init__(self, 
                 name: str = 'ETTm2', 
                 train_batch_size: int = 128, 
                 eval_batch_size: int = None,
                 num_workers: int = 4,
                 persistent_workers: bool = True,
                 horizon_length: int = 720,
                 history_length: int = 720,
                 split_proportions: List[float] = [0.6, 0.2, 0.2]
                ):
        super().__init__()
        self.name = name
        self.train_batch_size = train_batch_size
        self.eval_batch_size = train_batch_size
        if eval_batch_size is not None:
            self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.split_proportions = np.array(split_proportions).cumsum()
        
        assert self.split_proportions[-1] == 1, "Split proportions must sum up to 1"

    def prepare_data(self):
        LongHorizon.load(directory='./data', group='ETTm2')

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dataset = LongHorizonUnivariateDataset(name=self.name, split='train', 
                                                              split_start=0.0,
                                                              split_end=self.split_proportions[0],
                                                              horizon_length=self.horizon_length,
                                                              history_length = self.history_length)
            self.val_dataset = LongHorizonUnivariateDataset(name=self.name, split='val', 
                                                            split_start=self.split_proportions[0],
                                                            split_end=self.split_proportions[1],
                                                            horizon_length=self.horizon_length,
                                                            history_length = self.history_length)
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_dataset = LongHorizonUnivariateDataset(name=self.name, split='test', 
                                                             split_start=self.split_proportions[1],
                                                             split_end=self.split_proportions[2],
                                                             horizon_length=self.horizon_length,
                                                             history_length = self.history_length)
        if stage == "predict":
            self.predict_dataset = LongHorizonUnivariateDataset(name=self.name, split='test', 
                                                                split_start=self.split_proportions[1],
                                                                split_end=self.split_proportions[2],
                                                                horizon_length=self.horizon_length,
                                                                history_length = self.history_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, 
                          shuffle=True, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, 
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size,
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.eval_batch_size, 
                          shuffle=False, pin_memory=True, 
                          persistent_workers=self.persistent_workers,
                          num_workers=self.num_workers, collate_fn=collate_fn_flat_deal)
class ElectricityWithExogDataset(ElectricityUnivariateDataset):
    """Extended dataset with exogenous features support"""
    
    def __init__(self,
                 name: str,
                 split: str,
                 exog_features: List[str] = ['temperature', 'humidity'],
                 calendar_features: bool = True,
                 **kwargs):
        super().__init__(name, split, **kwargs)
        self.exog_features = exog_features
        self.calendar_features = calendar_features
        self.exog_df = None
        
        # Load exogenous data (aligned with time series)
        if exog_features:
            exog_path = f'./data/exogenous/{name.lower()}_weather.csv'
            try:
                self.exog_df = pd.read_csv(exog_path, parse_dates=['ds'])
                self.exog_df = self.exog_df.set_index('ds')
            except FileNotFoundError:
                print(f"Warning: Exogenous data file not found at {exog_path}")
                self.exog_features = []
    
    def _get_calendar_features(self, timestamps):
        """Generate calendar embeddings"""
        hour = timestamps.hour / 24.0  # [0, 1]
        dow = timestamps.dayofweek / 7.0  # [0, 1]
        month = (timestamps.month - 1) / 12.0  # [0, 1]
        is_weekend = (timestamps.dayofweek >= 5).astype(float)
        
        return np.stack([hour, dow, month, is_weekend], axis=-1)
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        
        # Need to get window_idx from parent class or calculate it
        # Assuming the parent class stores this information
        window_idx = index  # Adjust based on your parent class implementation
        
        # Get timestamps for this window
        window_times = self.series_timestamps[window_idx:window_idx + self.tot_window_len]
        
        # Add exogenous features
        if self.exog_features and self.exog_df is not None:
            try:
                exog = self.exog_df.loc[window_times][self.exog_features].values
                item['exog_history'] = exog[:self.history_length].astype(np.float32)
                item['exog_future'] = exog[self.history_length:].astype(np.float32)
            except KeyError:
                # If timestamps don't match, create zero arrays
                print(f"Warning: Timestamps mismatch for index {index}")
                item['exog_history'] = np.zeros((self.history_length, len(self.exog_features)), dtype=np.float32)
                item['exog_future'] = np.zeros((self.tot_window_len - self.history_length, len(self.exog_features)), dtype=np.float32)
        
        if self.calendar_features:
            item['calendar'] = self._get_calendar_features(window_times)
        
        return item