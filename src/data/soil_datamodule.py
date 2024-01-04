from typing import Any, Dict, Optional, Tuple

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
        train_period: Tuple[str, str] = ('2014-07-04', '2022-12-31'),
        val_period: Tuple[str, str] = ('2013-10-03', '2014-07-03'),
        test_period: Tuple[str, str] = ('2012-10-03', '2013-07-03'),
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

            # filter for time period
            amsr = amsr.sel(time=slice(self.hparams.test_period[0], self.hparams.train_period[1]))
            smos = smos.sel(time=slice(self.hparams.test_period[0], self.hparams.train_period[1]))
            ascat = ascat.sel(time=slice(self.hparams.test_period[0], self.hparams.train_period[1]))

            # rename to Latitude and Longitude
            ascat = ascat.rename({'latitude': 'Latitude', 'longitude':'Longitude'})

            # fill missing values with 1 (sea cells)
            amsr = amsr.fillna(1)
            smos = smos['SMOS_fill_smoothed'].fillna(1)
            ascat = ascat['ASCAT_fill_smooth'].fillna(1)

            # convert to numpy
            amsr_np = amsr.values
            smos_np = smos.values
            ascat_np = ascat.values

            # optional: normalize data

            # optional: add lookback window
            # X_sequences = []
            # Y_sequences = []
            # # create sequences
            # for i in range(len(ascat_np)):
            #     # check if there are enough past and future days for each i
            #     if i - self.hparams.days_before >= 0 and i + self.hparams.days_after < len(ascat_np):
            #         # create input sequences
            #         X_seq_1 = ascat_np[i-self.hparams.days_before:i+self.hparams.days_after]
            #         X_seq_2 = smos_np[i-self.hparams.days_before:i+self.hparams.days_after]
                    
            #         X_seq = np.concatenate((np.expand_dims(X_seq_1, axis=-1), np.expand_dims(X_seq_2, axis=-1)), axis=-1)
            #         X_sequences.append(X_seq)

            #         # Create output sequences
            #         Y_seq = amsr_np[i]
            #         Y_sequences.append(Y_seq)

            # # convert the lists to NumPy arrays
            # X = np.array(X_sequences)
            # Y = np.array(Y_sequences)

            # concatenate ascat and smos by adding a new dimension at the end (until we activate the lookback window)
            X = np.concatenate((np.expand_dims(ascat_np, axis=-1), np.expand_dims(smos_np, axis=-1)), axis=-1)
            Y = amsr_np

            # obtain the index positions of the specified periods
            train_index_pos = np.where(
                (amsr.time.values >= np.datetime64(self.hparams.train_period[0])) & 
                (amsr.time.values <= np.datetime64(self.hparams.train_period[1])))[0]
            val_index_pos = np.where(
                (amsr.time.values >= np.datetime64(self.hparams.val_period[0])) &
                (amsr.time.values <= np.datetime64(self.hparams.val_period[1])))[0]
            test_index_pos = np.where(
                (amsr.time.values >= np.datetime64(self.hparams.test_period[0])) &
                (amsr.time.values <= np.datetime64(self.hparams.test_period[1])))[0]

            # split into train, val and test and convert to torch tensors
            X_train = torch.from_numpy(X[train_index_pos]).float()
            Y_train = torch.from_numpy(Y[train_index_pos]).float()
            X_val = torch.from_numpy(X[val_index_pos]).float()
            Y_val = torch.from_numpy(Y[val_index_pos]).float()
            X_test = torch.from_numpy(X[test_index_pos]).float()
            Y_test = torch.from_numpy(Y[test_index_pos]).float()

            # create datasets
            self.data_train = SoilDataset(X_train, Y_train)
            self.data_val = SoilDataset(X_val, Y_val)
            self.data_test = SoilDataset(X_test, Y_test)


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

    def __init__(self, features, targets):
        """Initialize the dataset.

        :param df: The dataframe containing the soil moisture data.
        :param neighborhood: Whether to use the neighborhood data. Defaults to ``False``.
        """
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        """Return the length of the dataset.

        :return: The length of the dataset.
        """
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the item at the given index.

        :param idx: The index of the item to return.
        :return: The item at the given index.
        """
        x = self.features[idx]
        y = self.targets[idx]
        return x, y

if __name__ == "__main__":
    _ = SoilDataModule()
