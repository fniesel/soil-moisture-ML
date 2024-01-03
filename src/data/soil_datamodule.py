from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import pandas as pd


class SoilDataModule(LightningDataModule):
    """`LightningDataModule` for the Soil dataset.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        val_period: Tuple[str, str] = ('2013-10-03', '2014-07-03'),
        test_period: Tuple[str, str] = ('2012-10-03', '2013-07-03'),
        days_before: int = 3,
        days_after: int = 3,
        neighborhood: bool = True,
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
            # # load data
            # combined_dataset = pd.read_csv(self.hparams.data_dir + 'combined_dataset.csv')

            # # drop all cells that have below 1000 amsr and ascat observations
            # cells = combined_dataset.groupby(["latitude", "longitude"]).count()
            # cells = cells[cells["amsr"].values >= 1000]
            # cells = cells[cells["ascat"].values >= 1000]
            # cells = cells.reset_index()
            # filtered_dataset = combined_dataset.merge(cells[["latitude", "longitude"]], on=["latitude", "longitude"], how="inner")

            # # interpolate ascat_smoothed
            # interpolated_dataset = filtered_dataset.copy()
            # interpolated_dataset["ascat_smoothed"] = interpolated_dataset.groupby(["latitude", "longitude"])["ascat_smoothed"].apply(lambda x: x.interpolate(method='linear', limit_direction='both'))
         
            # # only keep relevant columns 
            # df = interpolated_dataset[["latitude", "longitude", "time", "year", "dayofyear", "amsr_smoothed", "ascat_smoothed"]]
            
            df = pd.read_csv(self.hparams.data_dir + 'filtered_interpolated_dataset.csv')
            # only keep relevant columns 
            df = df[["latitude", "longitude", "time", "year", "dayofyear", "amsr_smoothed", "ascat_smoothed"]]

            # include new features --> days before and after
            df.sort_values(by=['latitude', 'longitude', 'time'], inplace=True)
            for i in range(self.hparams.days_before):
                df[f"{-(i+1)}"] = df['ascat_smoothed'].shift(i+1)
            for i in range(self.hparams.days_after):
                df[f"{i+1}"] = df['ascat_smoothed'].shift(-(i+1))
            # drop the first and last days of each time series (would contain NaN values)
            df = df[~df.time.isin(df.groupby(['latitude', 'longitude']).head(self.hparams.days_before).time.unique())]
            df = df[~df.time.isin(df.groupby(['latitude', 'longitude']).tail(self.hparams.days_after).time.unique())]

            # add helper columns that specify the neighborhood coordinates
            if self.hparams.neighborhood:
                df['top_lat'] = df['latitude'] + 0.25
                df['bottom_lat'] = df['latitude'] - 0.25
                df['left_lon'] = df['longitude'] - 0.25
                df['right_lon'] = df['longitude'] + 0.25

                # merge the df with itself to get the neighboring ascat values
                df_helper = df[['time', 'latitude', 'longitude', 'ascat_smoothed']]
                df_helper.columns = ['time', 'top_lat', 'longitude', 'ascat_smoothed_top']
                df = pd.merge(df, df_helper, on=['time', 'top_lat', 'longitude'], how='left')
                df_helper.columns = ['time', 'bottom_lat', 'longitude', 'ascat_smoothed_bottom']
                df = pd.merge(df, df_helper, on=['time', 'bottom_lat', 'longitude'], how='left')
                df_helper.columns = ['time', 'latitude', 'left_lon', 'ascat_smoothed_left']
                df = pd.merge(df, df_helper, on=['time', 'latitude', 'left_lon'], how='left')
                df_helper.columns = ['time', 'latitude', 'right_lon', 'ascat_smoothed_right']
                df = pd.merge(df, df_helper, on=['time', 'latitude', 'right_lon'], how='left')

                # drop helper columns
                df.drop(columns=['top_lat', 'bottom_lat', 'left_lon', 'right_lon'], inplace=True)

                # drop cells that have NaN values (cells at the border of the dataset)
                df = df.dropna(subset=['ascat_smoothed_top', 'ascat_smoothed_bottom', 'ascat_smoothed_left', 'ascat_smoothed_right'])

            # drop rows with missing amsr or ascat values
            df = df.dropna(subset=['amsr_smoothed'])

            # set time as index
            df.set_index('time', inplace=True)

            # split data
            val = df[(df.index >= self.hparams.val_period[0]) & (df.index <= self.hparams.val_period[1])]
            test = df[(df.index >= self.hparams.test_period[0]) & (df.index <= self.hparams.test_period[1])]
            train = df[(df.index < self.hparams.test_period[0]) | (df.index > self.hparams.val_period[1])]

            features_train = torch.from_numpy(train.drop(columns='amsr_smoothed').values).float()
            features_val = torch.from_numpy(val.drop(columns='amsr_smoothed').values).float()
            features_test = torch.from_numpy(test.drop(columns='amsr_smoothed').values).float()

            # todo: maybe normalize data
            
            labels_train = torch.from_numpy(train['amsr_smoothed'].values).float()
            labels_val = torch.from_numpy(val['amsr_smoothed'].values).float()
            labels_test = torch.from_numpy(test['amsr_smoothed'].values).float()

            # create datasets
            self.data_train = SoilDataset(features_train, labels_train)
            self.data_val = SoilDataset(features_val, labels_val)
            self.data_test = SoilDataset(features_test, labels_test)

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
