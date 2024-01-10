from typing import Any, Dict, Optional, Tuple, List

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import pandas as pd
import xarray as xr
import numpy as np


class SoilDataModule(LightningDataModule):
    """`LightningDataModule` for the Soil dataset.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "/p/scratch/share/sivaprasad1/niesel1/",
        train_period: Tuple[str, str] = ('2014-07-04', '2022-12-21'),
        val_period: Tuple[str, str] = ('2013-10-03', '2014-07-03'),
        test_period: Tuple[str, str] = ('2012-10-03', '2013-07-03'),
        features: List[str] = ['ascat', 'ascat_anomaly'],
        target: str = 'amsr_anomaly',
        days_before: int = 3,
        days_after: int = 3,
        min_clip_value: float = 0,
        max_clip_value: float = 0.95,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `SoilDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param val_period: The period to use for validation. Defaults to `('2013-10-03', '2014-07-03')`.
        :param test_period: The period to use for testing. Defaults to `('2012-10-03', '2013-07-03')`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            
            # load data
            amsr = xr.open_dataset(f"{self.hparams.data_dir}/AMSR_cp_smooth_final_sg7_3.nc")
            smos = xr.open_dataset(f"{self.hparams.data_dir}/SMOS_cp_smooth_final.nc")
            ascat = xr.open_dataset(f"{self.hparams.data_dir}/ASCAT_cp_smooth.nc")   

            # convert amsr to volumetric soil moisture
            amsr = amsr['AMSR_fill_smooth']/100

            # clip values to min and max soil moisture values (occurs due to smoothing)
            amsr = amsr.clip(min=self.hparams.min_clip_value, max=self.hparams.max_clip_value)
            smos = smos.clip(min=self.hparams.min_clip_value, max=self.hparams.max_clip_value)
            ascat = ascat.clip(min=self.hparams.min_clip_value, max=self.hparams.max_clip_value)

            # convert time to same format in each dataset
            amsr['time'] = pd.to_datetime(amsr['time'].values)
            ascat['time'] = pd.to_datetime(ascat['time'].values)
            smos['time'] = pd.to_datetime(smos['time'].values)

            # filter for time period (include the days before and after for context)
            amsr = amsr.sel(time=slice(
                np.datetime64(self.hparams.test_period[0])-self.hparams.days_before, 
                np.datetime64(self.hparams.train_period[1])+self.hparams.days_after))
            smos = smos.sel(time=slice(
                np.datetime64(self.hparams.test_period[0])-self.hparams.days_before, 
                np.datetime64(self.hparams.train_period[1])+self.hparams.days_after))
            ascat = ascat.sel(time=slice(
                np.datetime64(self.hparams.test_period[0])-self.hparams.days_before,
                np.datetime64(self.hparams.train_period[1])+self.hparams.days_after))

            # rename to Latitude and Longitude
            ascat = ascat.rename({'latitude': 'Latitude', 'longitude':'Longitude'})

            # fill missing values with 1 (sea cells)
            amsr = amsr.fillna(1)
            smos = smos['SMOS_fill_smoothed'].fillna(1)
            ascat = ascat['ASCAT_fill_smooth'].fillna(1)

            # compute trend given the training data
            amsr_trend = amsr.sel(time=slice(
                np.datetime64(self.hparams.train_period[0]), 
                np.datetime64(self.hparams.train_period[1]))).groupby('time.dayofyear').mean(dim='time')
            smos_trend = smos.sel(time=slice(
                np.datetime64(self.hparams.train_period[0]), 
                np.datetime64(self.hparams.train_period[1]))).groupby('time.dayofyear').mean(dim='time')
            ascat_trend = ascat.sel(time=slice(
                np.datetime64(self.hparams.train_period[0]), 
                np.datetime64(self.hparams.train_period[1]))).groupby('time.dayofyear').mean(dim='time')

            # compute anomaly
            amsr_anomaly = amsr.groupby('time.dayofyear') - amsr_trend
            smos_anomaly = smos.groupby('time.dayofyear') - smos_trend
            ascat_anomaly = ascat.groupby('time.dayofyear') - ascat_trend

            # create Y w.r.t. the target variable
            if self.hparams.target == 'amsr':
                Y = amsr.values
            elif self.hparams.target == 'amsr_anomaly':
                Y = amsr_anomaly.values

            # create X w.r.t. the feature variables
            feature_list = []
            if 'ascat' in self.hparams.features:
                feature_list.append(np.expand_dims(ascat.values, axis=-1))
            if 'ascat_anomaly' in self.hparams.features:
                feature_list.append(np.expand_dims(ascat_anomaly.values, axis=-1))
            if 'smos' in self.hparams.features:
                feature_list.append(np.expand_dims(smos.values, axis=-1))
            if 'smos_anomaly' in self.hparams.features:
                feature_list.append(np.expand_dims(smos_anomaly.values, axis=-1))
            X = np.concatenate(feature_list, axis=-1)

            # optional: normalize data

            # obtain the index positions of the specified periods (include the days before and after for context)
            train_index_pos = np.where(
                (amsr.time.values >= np.datetime64(self.hparams.train_period[0])-self.hparams.days_before) & 
                (amsr.time.values <= np.datetime64(self.hparams.train_period[1])+self.hparams.days_after))[0]
            val_index_pos = np.where(
                (amsr.time.values >= np.datetime64(self.hparams.val_period[0])-self.hparams.days_before) &
                (amsr.time.values <= np.datetime64(self.hparams.val_period[1])+self.hparams.days_after))[0]
            test_index_pos = np.where(
                (amsr.time.values >= np.datetime64(self.hparams.test_period[0])-self.hparams.days_before) &
                (amsr.time.values <= np.datetime64(self.hparams.test_period[1])+self.hparams.days_after))[0]

            # split into train, val and test and convert to torch tensors
            X_train = torch.from_numpy(X[train_index_pos]).float()
            Y_train = torch.from_numpy(Y[train_index_pos]).float()
            X_val = torch.from_numpy(X[val_index_pos]).float()
            Y_val = torch.from_numpy(Y[val_index_pos]).float()
            X_test = torch.from_numpy(X[test_index_pos]).float()
            Y_test = torch.from_numpy(Y[test_index_pos]).float()

            # create datasets
            self.data_train = SoilDataset(X_train, Y_train, self.hparams.days_before, self.hparams.days_after)
            self.data_val = SoilDataset(X_val, Y_val, self.hparams.days_before, self.hparams.days_after)
            self.data_test = SoilDataset(X_test, Y_test, self.hparams.days_before, self.hparams.days_after)


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


class SoilDataset(Dataset):
    """Dataset class for soil moisture data."""

    def __init__(self, features, targets, days_before, days_after):
        """Initialize the dataset.

        :param features: Input data with first dimension corresponding to days.
        :param targets: Output data with first dimension corresponding to days.
        :param days_before: Number of days before the target day to include in x.
        :param days_after: Number of days after the target day to include in x.
        """
        self.features = features
        self.targets = targets
        self.days_before = days_before
        self.days_after = days_after

    def __len__(self) -> int:
        """Return the length of the dataset.

        :return: The length of the dataset.
        """
        # subtract days_before and days_after because we can't use those days as targets
        return len(self.targets) - self.days_before - self.days_after

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the item at the given index.

        :param idx: The index of the item to return.
        :return: The item at the given index.
        """
        # get the input and output data for the given index (w.r.t the days before and after)
        x = self.features[idx:idx+self.days_before+self.days_after+1]
        y = self.targets[idx+self.days_before]
        return x, y

if __name__ == "__main__":
    _ = SoilDataModule()
