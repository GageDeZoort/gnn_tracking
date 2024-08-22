import itertools
import os
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import RandomSampler
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from gnn_tracking.preprocessing.point_cloud_builder import PointCloudBuilder
from gnn_tracking.utils.log import logger

DEFAULT_FEATURES = (
    "r",
    "phi",
    "z",
    "eta_rz",
    "u",
    "v",
    "charge_frac",
    "leta",
    "lphi",
    "lx",
    "ly",
    "lz",
    "geta",
    "gphi",
)


# noinspection PyAbstractClass
class TrackingDataset(Dataset):
    def __init__(
        self,
        in_dir: str | os.PathLike | list[str] | list[os.PathLike],
        *,
        start=0,
        stop=None,
        sector: int | None = None,
        point_cloud_builder: PointCloudBuilder | None,
        feature_subset_names: list[str] | None = None,
    ):
        """Dataset for tracking applications

        Args:
            in_dir: Directory or list of directories containing the data files
            start: Index of the first file to be considered (with files from the
                in dirs considered in order)
            stop: Index of the last file to be considered
            sector: If not None, only files with this sector number will be considered
        """
        super().__init__()
        self.point_cloud_builder = point_cloud_builder
        self._processed_paths = self._get_paths(
            in_dir, start=start, stop=stop, sector=sector
        )
        self.file_number = 0
        self.prev_file_number = -1
        self.sector_results = []
        if feature_subset_names is not None:
            self.feature_subset = [
                DEFAULT_FEATURES.index(name) for name in feature_subset_names
            ]
        else:
            self.feature_subset = None

    def _get_paths(
        self,
        in_dir: str | os.PathLike | list[str] | list[os.PathLike],
        *,
        start=0,
        stop: int | None = None,
        sector: int | None = None,
    ) -> list[Path]:
        """Collect all paths that should be in this dataset."""
        if start == stop:
            return []

        if self.point_cloud_builder is not None:
            in_dir = Path(self.point_cloud_builder.indir)
            glob = "*-hits.csv.gz"  # avoid overcounting
        else:
            glob = "*.pt" if sector is None else f"*_s{sector}.pt"

        if not isinstance(in_dir, list):
            in_dir = [in_dir]
        for d in in_dir:
            if not Path(d).exists():
                msg = f"Directory {d} does not exist."
                raise FileNotFoundError(msg)

        available_files = sorted(
            itertools.chain.from_iterable([Path(d).glob(glob) for d in in_dir])
        )

        if stop is not None and stop > len(available_files):
            # to avoid tracking wrong hyperparameters
            msg = f"stop={stop} is larger than the number of files  ({len(available_files)})"
            raise ValueError(msg)
        considered_files = available_files[start:stop]
        logger.info(
            "DataLoader will load %d graphs (out of %d available).",
            len(considered_files),
            len(available_files),
        )
        logger.debug(
            "First graph is %s, last graph is %s",
            considered_files[0],
            considered_files[-1],
        )
        return considered_files

    def len(self) -> int:
        if self.point_cloud_builder is None or self.point_cloud_builder.n_sectors == 1:
            return len(self._processed_paths)

        return len(self._processed_paths) * self.point_cloud_builder.n_sectors

    def get(self, idx: int) -> Data:
        # slightly funky logic to load each sector on the fly without re-processing each file
        if self.point_cloud_builder is None:
            data = torch.load(self._processed_paths[idx])
            if self.feature_subset is not None:
                if data.x.shape[1] < len(self.feature_subset):
                    msg = f"error in file {self._processed_paths[idx]}"
                    raise ValueError(msg)
                data.x = data.x[:, self.feature_subset]
            return data

        if self.point_cloud_builder.n_sectors == 1:
            return self.point_cloud_builder.process(idx, idx + 1)

        self.file_number = idx // self.point_cloud_builder.n_sectors
        if self.file_number != self.prev_file_number:
            self.sector_results = self.point_cloud_builder.process(
                self.file_number, self.file_number + 1
            )
        self.prev_file_number = self.file_number
        return self.sector_results[
            idx - self.file_number * self.point_cloud_builder.n_sectors
        ]


class TrackingDataModule(LightningDataModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        *,
        identifier: str,  # noqa: ARG002
        train: dict | None = None,
        val: dict | None = None,
        test: dict | None = None,
        predict: dict | None = None,
        cpus: int = 1,
        builder_params: dict | None = None,
        feature_subset_names: list[str] | None = None,
    ):
        """This subclass of `LightningDataModule` configures all data for the
        ML pipeline.

        Args:
            identifier: Identifier of the dataset (e.g., `graph_v5`)
            train: Config dictionary for training data (see below)
            val: Config dictionary for validation data (see below)
            test: Config dictionary for test data (see below)
            cpus: Number of CPUs to use for loading data.


        The following keys are available for each config dictionary:

        - `dirs`: List of dirs to load from (required)
        - `start=0`: Index of first file to load
        - `stop=None`: Index of last file to load
        - `sector=None`: Sector to load from (if None, load all sectors)
        - `batch_size=1`: Batch size

        Training has the following additional keys:

        - `sample_size=None`: Number of samples to load for each epoch
            (if None, load all samples)
        """
        self.save_hyperparameters()
        super().__init__()
        self._configs = {
            "train": self._fix_datatypes(train),
            "val": self._fix_datatypes(val),
            "test": self._fix_datatypes(test),
            "predict": self._fix_datatypes(predict),
        }
        self._datasets = {}
        self._cpus = cpus
        self.builder_params = builder_params
        self.feature_subset_names = feature_subset_names

    @property
    def datasets(self) -> dict[str, TrackingDataset]:
        if not self._datasets:
            logger.error(
                "Datasets have not been loaded yet. Make sure to call the setup method."
            )
        return self._datasets

    @staticmethod
    def _fix_datatypes(dct: dict[str, Any] | None) -> dict[str, Any] | None:
        """Fix datatypes of config dictionary.
        This is necessary because when configuring values from the command line,
        all values might be strings.
        """
        if dct is None:
            return {}
        for key in ["start", "stop", "sector", "batch_size", "sample_size"]:
            if key in dct:
                dct[key] = int(dct[key])
        return dct

    def _get_dataset(self, key: str) -> TrackingDataset:
        config = self._configs[key]

        if self.builder_params is not None:
            in_dir = self.builder_params["indir"]

        else:
            in_dir = config["dirs"]
        config = self._configs[key]
        if not config:
            msg = f"DataLoaderConfig for key {key} is None."
            raise ValueError(msg)
        point_cloud_builder = None
        if self.builder_params is not None:
            point_cloud_builder = PointCloudBuilder(**self.builder_params)
        return TrackingDataset(
            in_dir=in_dir,
            start=config.get("start", 0),
            stop=config.get("stop", None),
            sector=config.get("sector", None),
            point_cloud_builder=point_cloud_builder,  # Pass builder
            feature_subset_names=self.feature_subset_names,
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._datasets["train"] = self._get_dataset("train")
            self.setup("validate")
        elif stage == "validate":
            self._datasets["val"] = self._get_dataset("val")
        elif stage == "test":
            self._datasets["test"] = self._get_dataset("test")
        elif stage == "predict":
            self._datasets["predict"] = self._get_dataset("predict")
        else:
            _ = f"Unknown stage '{stage}'"
            raise ValueError(_)

    def _get_dataloader(self, key: str) -> DataLoader:
        sampler = None
        dataset = self._datasets[key]
        n_samples = len(dataset)
        if key == "train" and len(self._datasets[key]):
            if "max_sample_size" in self._configs[key]:
                msg = "max_sample_size has been replaced by sample_size"
                raise ValueError(msg)
            n_samples = self._configs[key].get("sample_size", len(dataset))
            sampler = RandomSampler(
                self._datasets[key],
                replacement=n_samples > len(dataset),
                num_samples=n_samples,
            )
        return DataLoader(
            dataset,
            batch_size=self._configs[key].get("batch_size", 1),
            num_workers=max(1, min(n_samples, self._cpus)),
            sampler=sampler,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")

    def predict_dataloader(self):
        return self._get_dataloader("predict")


class TestTrackingDataModule(LightningDataModule):
    def __init__(self, graphs: list[Data]):
        """Version of `TrackingDataLoader` only used for testing purposes."""
        super().__init__()
        self.graphs = graphs
        self.datasets = {
            "train": [graphs[0]],
            "val": [graphs[1]],
            "test": [graphs[1]],
        }

    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(self.datasets["train"])

    def val_dataloader(self):
        return DataLoader(self.datasets["val"])

    def test_dataloader(self):
        return DataLoader(self.datasets["test"])
