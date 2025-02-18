"""Build point clouds from the input data files."""

import collections
import itertools
import logging
import traceback
from collections import defaultdict
from pathlib import Path, PurePath
from typing import Any, Literal

import awkward as ak
import numpy as np
import pandas as pd
import torch
import uproot
from torch_geometric.data import Data

import gnn_tracking.preprocessing.exatrkx_cell_features as ecf
from gnn_tracking.utils.log import get_logger

pd.options.mode.chained_assignment = None  # default='warn'


def get_truth_edge_index(pids: np.ndarray) -> np.ndarray:
    # Step 1: Store indices by particle ID in a dictionary

    particle_indices = defaultdict(list)
    for idx, pid in enumerate(pids):
        if pid != 0:  # Skip particle ID 0
            particle_indices[pid].append(idx)
    # Step 2: Generate edges directly
    edges = []
    for indices in particle_indices.values():
        if len(indices) < 2:
            continue
        edges.extend(itertools.combinations(indices, 2))
    return np.array(edges).T


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
_DEFAULT_FEATURE_SCALE = tuple(1 for _ in DEFAULT_FEATURES)


# TODO: In need of refactoring: load_point_clouds should be factored out (this should
#   only be used for building the graphs), and the parsing of the filenames should be
#   done with the function that is also used in build_point_clouds
#   Split up in feature building and sectorization?
#   Class should be refactored as well: Most methods, attributes are private, many are
#   static. Refactoring to be subclass of HyperparametersMixin would allow to easily
#   save all hyperparameters in yaml output file which could be useful for versioning

# For timing performance, the costly functions are loading the data (36%),
# appending features (22%), and getting edges (25%).


class PointCloudBuilder:
    def __init__(
        self,
        *,
        outdir: str | PurePath,
        indir: str | PurePath,
        detector_config: PurePath,
        n_sectors: int,
        redo: bool = True,
        pixel_only: bool = True,
        sector_di: float = 0.0001,
        sector_ds: float = 1.1,
        measurement_mode: bool = False,
        thld: float = 0.5,
        remove_noise: bool = False,
        write_output: bool = True,
        log_level=logging.INFO,
        collect_data: bool = False,
        feature_names: tuple = DEFAULT_FEATURES,
        feature_scale: tuple = _DEFAULT_FEATURE_SCALE,
        add_true_edges: bool = False,
        return_data: bool = False,
        data_type: str = Literal["TrackML", "CMS_MC", "CMS_Run3"],

    ):
        """Build point clouds, that is, read the input data files and convert them
        to pytorch geometric data objects (without any edges yet).

        Args:
            outdir: Directory for the output files
            indir: Directory for the input files
            detector_config: Path to the detector configuration file
            n_sectors: Total number of sectors
            redo: Re-compute the point cloud even if it is found
            pixel_only: Construct tracks only from pixel layers
            sector_di: The intercept offset for the extended sector
            sector_ds: The slope offset for the extended sector
            measurement_mode: Produce statistics about the sectorization
            thld: Threshold pt for measurements
            remove_noise: Remove hits with particle_id==0
            write_output: Store the point clouds in a torch .pt file
            log_level: Specify INFO (0) or DEBUG (>0)
            collect_data: Collect data in memory
            feature_names: Names of features to add
            feature_scale: Scale of features
            add_true_edges: Add true edges to the point cloud
        """
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.initial_outdir = self.outdir
        self.indir = Path(indir)
        self.n_sectors = n_sectors
        self.redo = redo
        self.pixel_only = pixel_only
        self.sector_di = sector_di
        self.sector_ds = sector_ds
        self.measurement_mode = measurement_mode
        self.thld = thld
        self.stats = {}
        self.remove_noise = remove_noise
        self.measurements: list[dict[str, Any]] = []
        self.write_output = write_output
        self.feature_names = list(feature_names)
        self.feature_scale = list(feature_scale)
        assert len(self.feature_names) == len(self.feature_scale)
        self.return_data = return_data
        self.data_type = data_type

        suffix = "-hits.csv.gz"
        self.ntuple = ""
        self.prefixes: list[Path] = []
        #: Does an output file for a given key exist?
        self.exists: dict[str, bool] = {}
        outfiles = [child.name for child in self.outdir.iterdir()]
        # Sort the files to keep unit tests fixed on different platforms
        self.infiles = sorted(self.indir.iterdir())
        if self.data_type == "TrackML":
            for p in self.infiles:
                if p.name.endswith(suffix):
                    prefix = p.name.replace(suffix, "")
                    evtid = int(prefix[-9:])
                    for s in range(self.n_sectors):
                        key = f"data{evtid}_s{s}.pt"
                        self.exists[key] = key in outfiles
                    self.prefixes.append(self.indir / prefix)

        self.data_list: list[Data] = []
        self.logger = get_logger("PointCloudBuilder", level=log_level)
        self._collect_data = collect_data
        self.add_true_edges = add_true_edges
        self._detector = ecf.load_detector(Path(detector_config))[1]

    @staticmethod
    def calc_eta(r: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Compute pseudorapidity (spatial)."""
        theta = np.arctan2(r, z)
        return -np.log(np.tan(theta / 2.0))

    def restrict_to_subdetectors(self, hits: pd.DataFrame) -> pd.DataFrame:
        """Rename (volume, layer) pairs with an integer label. If only pixel det, subset data"""
        pixel_barrel = [(8, 2), (8, 4), (8, 6), (8, 8)]
        pixel_LEC = [(7, 14), (7, 12), (7, 10), (7, 8), (7, 6), (7, 4), (7, 2)]
        pixel_REC = [(9, 2), (9, 4), (9, 6), (9, 8), (9, 10), (9, 12), (9, 14)]
        if self.pixel_only:
            available_allowed_layers = sorted(pixel_barrel + pixel_REC + pixel_LEC)
        else:
            available_allowed_layers = (
                hits[["volume_id", "layer_id"]]
                .value_counts()
                .reset_index()
                .sort_values(by=["volume_id", "layer_id"])
            )
            available_allowed_layers = [
                tuple(x)
                for x in available_allowed_layers[["volume_id", "layer_id"]].to_numpy()
            ]

        new_layer_ids = dict(
            zip(available_allowed_layers, range(len(available_allowed_layers)))
        )

        hits["layer"] = pd.Series(list(zip(hits["volume_id"], hits["layer_id"]))).map(
            new_layer_ids
        )

        return hits

    def append_features(
        self,
        hits: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add additional features to the hits dataframe and return it."""

        hits["r"] = np.sqrt(hits.x**2 + hits.y**2)
        hits["phi"] = np.arctan2(hits.y, hits.x)
        hits["eta_rz"] = self.calc_eta(hits.r, hits.z)
        hits["u"] = hits["x"] / (hits["x"] ** 2 + hits["y"] ** 2)
        hits["v"] = hits["y"] / (hits["x"] ** 2 + hits["y"] ** 2)
        return hits

    def append_cell_features(
        self, hits: pd.DataFrame, cells: pd.DataFrame
    ) -> pd.DataFrame:
        # add in channel-specific info
        cells_agg = cells.groupby(["hit_id"]).agg(
            charge_sum=pd.NamedAgg(column="value", aggfunc="sum"),
            channel_counts=pd.NamedAgg(column="value", aggfunc="size"),
        )
        cells_agg["charge_frac"] = cells_agg.charge_sum / cells_agg.channel_counts
        hits = pd.merge(hits, cells_agg, on="hit_id", how="left")

        return ecf.augment_hit_features(hits, cells, detector_proc=self._detector)

    def sector_hits(
        self, hits: pd.DataFrame, sector_id: int, particle_id_counts: dict[int, int]
    ) -> pd.DataFrame:
        """Break an event into (optionally) extended sectors."""

        if self.n_sectors == 1:
            hits["sector"] = 0
            return hits

        # build sectors in each 2*np.pi/self.n_sectors window
        theta = np.pi / self.n_sectors
        slope = np.arctan(theta)
        hits["ur"] = hits["u"] * np.cos(2 * sector_id * theta) - hits["v"] * np.sin(
            2 * sector_id * theta
        )
        hits["vr"] = hits["u"] * np.sin(2 * sector_id * theta) + hits["v"] * np.cos(
            2 * sector_id * theta
        )
        sector = hits[
            ((hits.vr > -slope * hits.ur) & (hits.vr < slope * hits.ur) & (hits.ur > 0))
        ]

        # assign when the majority of the particle's hits are in a sector
        particle_id_sectors = collections.defaultdict(lambda: -1)
        for pid in np.unique(sector.particle_id.to_numpy()):
            if pid == 0:
                continue
            hits_in_sector = len(sector[sector.particle_id == pid])
            hits_for_pid = particle_id_counts[particle_id_counts["particle_id"] == pid]
            if (hits_in_sector / len(hits_for_pid)) >= 0.5:
                particle_id_sectors[pid] = sector_id

        lower_bound = -self.sector_ds * slope * hits.ur - self.sector_di
        upper_bound = self.sector_ds * slope * hits.ur + self.sector_di
        extended_sector = hits[
            ((hits.vr > lower_bound) & (hits.vr < upper_bound) & (hits.ur > 0))
        ]

        extended_sector["sector"] = extended_sector["particle_id"].map(
            particle_id_sectors
        )

        measurements = {}
        if self.measurement_mode:
            measurements["n_hits"] = len(sector)
            measurements["n_hits_ext"] = len(extended_sector)
            if len(sector) > 0:
                measurements["n_hits_ratio"] = len(extended_sector) / len(sector)
            else:
                measurements["n_hits_ratio"] = 0

            measurements["n_unique_pids"] = len(
                np.unique(extended_sector.particle_id.to_numpy())
            )

            majority_contained = []
            for pid in np.unique(extended_sector.particle_id.to_numpy()):
                if pid == 0:
                    continue
                group = hits[hits.particle_id == pid]
                in_sector = (
                    (group.vr < slope * group.ur)
                    & (group.vr > -slope * group.ur)
                    & (group.pt >= self.thld)
                )
                n_total = len(
                    particle_id_counts[particle_id_counts["particle_id"] == pid]
                )
                if sum(in_sector) / n_total < 0.5:
                    continue

                in_ext_sector = (
                    (group.vr < (self.sector_ds * slope * group.ur + self.sector_di))
                    & (group.vr > (-self.sector_ds * slope * group.ur - self.sector_di))
                    & (group.pt > self.thld)
                )
                majority_contained.append(sum(in_ext_sector) == n_total)

            def zero_div(x, y):
                return x / y if y != 0 else 0

            efficiency = zero_div(sum(majority_contained), len(majority_contained))
            measurements["majority_contained"] = efficiency
            self.measurements.append(measurements)

        return extended_sector

    def _get_edge_index(self, particle_id: np.ndarray) -> torch.Tensor:
        if self.add_true_edges:
            edges = torch.tensor(get_truth_edge_index(particle_id)).long()
        else:
            edges = torch.zeros((2, 0)).long()
        return edges

    def to_pyg_data(self, hits: pd.DataFrame) -> Data:
        """Build the output data structure"""
        return Data(
            x=torch.tensor(
                hits[self.feature_names].to_numpy() / self.feature_scale,
                dtype=torch.float32,
            ),
            edge_index=self._get_edge_index(hits["particle_id"].values),
            y=torch.zeros(0).float(),
            layer=torch.tensor(hits.layer_id.values).long(),
            particle_id=torch.tensor(hits["particle_id"].values).long(),
            pt=torch.tensor(hits["pt"].values).float(),
            reconstructable=torch.tensor(hits["reconstructable"].values).long(),
            sector=torch.tensor(hits["sector"].values).long(),
            eta=torch.tensor(hits["eta_pt"].values).float(),
            n_hits=torch.tensor(hits["n_hits"].values).long(),
            n_layers_hit=torch.tensor(hits["n_layers_hit"].values).long(),
        )

    def get_measurements(self) -> dict[str, float]:
        measurements = pd.DataFrame(self.measurements)
        means = measurements.mean()
        stds = measurements.std()
        output = {}
        for var in means.index:
            output[var] = means[var]
            output[var + "_err"] = stds[var]
        return output

    def process(
        self,
        start: int | None = 0,
        stop: int | None = None,
        ignore_loading_errors=False,
    ):
        """Process input files from self.input_files and write output files to
        self.output_files

        Args:
            start: index of first file to process
            stop: index of last file to process (or None). Can be higher than total
                number of files.
            ignore_loading_errors: if True, ignore errors when loading event
        Returns:

        """
        for i in range(start, stop):
            self.logger.debug("Processing %s", i)

            try:
                if self.data_type == "TrackML":
                    f = self.prefixes[i]
                    evtid = int(f.name[-9:])
                    hits = self.read_trackml(f)
                elif self.data_type == "CMS_MC":
                    evtid = i
                    hits = self.read_cms_mc(i)
            except Exception:
                if ignore_loading_errors:
                    self.logger.error("Error loading event %d", evtid)
                    self.logger.error(traceback.format_exc())
                    continue
                raise

            pid_layer_count = (
                hits.groupby("particle_id")
                .agg(
                    n_hits=("particle_id", "size"), n_layers_hit=("layer_id", "nunique")
                )
                .reset_index()
            )
            # is the second condition necessary...
            hits = hits.merge(pid_layer_count, on="particle_id", how="left")
            hits["reconstructable"] = (hits["n_layers_hit"] >= 3) & (
                hits["particle_id"] > 0
            )
            n_particles = len(np.unique(hits.particle_id.to_numpy()))
            n_hits = len(hits)
            n_noise = len(hits[hits.particle_id == 0])
            n_sector_hits = 0  # total quantities appearing in sectored graph
            n_sector_particles = 0
            sector_list = []

            for s in range(self.n_sectors):
                name = f"data{evtid}_s{s}.pt"
                # not implemented with CMS yet
                if self.data_type == "TrackML" and self.exists[name] and not self.redo:
                    if self._collect_data:
                        data = torch.load(self.outdir / name)
                        # this append can slow the code down too much
                        self.data_list.append(data)
                    self.logger.debug("skipping %s", name)
                    continue
                sector = self.sector_hits(
                    hits,
                    s,
                    particle_id_counts=pid_layer_count[["particle_id", "n_hits"]],
                )
                n_sector_hits += len(sector)
                n_sector_particles += len(np.unique(sector.particle_id.to_numpy()))
                sector = self.to_pyg_data(sector)
                if self.return_data and self.n_sectors > 1:
                    sector_list.append(sector)
                outfile = self.outdir / name
                if self.write_output:
                    torch.save(sector, outfile)
                if self._collect_data:
                    self.data_list.append(sector)
                self.logger.debug("wrote %s", outfile)

            self.stats[evtid] = {
                "n_hits": n_hits,
                "n_particles": n_particles,
                "n_noise": n_noise,
                "n_sector_hits": n_sector_hits,
                "n_sector_particles": n_sector_particles,
            }

            for var in list(locals().keys()):  # Use list() to avoid RuntimeError
                del locals()[var]  # Deleting variable from local scope

        self.logger.debug("Output statistics: %s", self.stats[evtid])
        if self.measurement_mode:
            measurements = pd.DataFrame(self.measurements)
            means = measurements.mean()
            stds = measurements.std()
            for var in stds.index:
                _ = f"{var}: {means[var]:.4f}+/-{stds[var]:.4f}"
                self.logger.debug(_)

    # this speeds up the code slightly, if need more capability, use from trackml.dataset import load_event
    def read_trackml(self, f):
        f = str(f)
        suffix = ".csv.gz"
        cells = pd.read_csv(f + "-cells" + suffix, header=0, index_col=False)
        hits = pd.read_csv(f + "-hits" + suffix, header=0, index_col=False)
        truth = pd.read_csv(f + "-truth" + suffix, header=0, index_col=False)
        particles = pd.read_csv(f + "-particles" + suffix, header=0, index_col=False)

        hits = self.restrict_to_subdetectors(hits)
        hits = hits.dropna(subset="layer")

        cells = cells[cells.hit_id.isin(hits.hit_id)].copy()
        # handle noise
        truth_noise = truth[["hit_id", "particle_id"]][truth.particle_id == 0]
        truth_noise["pt"] = 0
        particles["pt"] = np.sqrt(particles.px**2 + particles.py**2)
        particles["eta_pt"] = self.calc_eta(particles.pt, particles.pz)
        truth = truth[["hit_id", "particle_id"]].merge(
            particles[["particle_id", "pt", "eta_pt", "q", "vx", "vy"]],
            on="particle_id",
        )

        # optionally add noise
        if not self.remove_noise:
            truth = pd.concat([truth, truth_noise])

        hits = self.append_cell_features(hits, cells)
        hits = self.append_features(hits)

        hits = hits.merge(truth[["hit_id", "particle_id", "pt", "eta_pt"]], on="hit_id")

        # append volume labels as one-hot features to X
        volume_labels = ["V7", "V8", "V9", "V12", "V13", "V14", "V16", "V17", "V18"]
        for v in volume_labels:
            hits[v] = (hits.volume_id == int(v[1:])).astype(int)

        return hits

    def read_cms_mc(self, evt_num):
        # the event numbering in each file starts from zero
        file_evt_num = evt_num - int(evt_num / 1000) * 1000
        if evt_num % 1000 == 0 or self.ntuple == "":
            self.ntuple, self.key, self.hit_key, self.cell_key = (
                self.load_new_cms_mc_file(evt_num)
            )
            subdir = Path(f"part_{int(evt_num/1000)}")
            self.outdir = self.initial_outdir / subdir
            # Create the directory if it doesn't exist
        self.outdir.mkdir(parents=True, exist_ok=True)
        hits = ak.to_dataframe(
            self.ntuple[self.key][self.hit_key].arrays(
                entry_start=file_evt_num, entry_stop=file_evt_num + 1
            )
        )
        cells = ak.to_dataframe(
            self.ntuple[self.key][self.cell_key].arrays(
                entry_start=file_evt_num, entry_stop=file_evt_num + 1
            )
        )
        cells = cells.rename({"charge_value": "value"}, axis=1)
        assert len(hits) > 0, f"the hits in event {evt_num} are empty "
        hits = hits.rename({"sim_pt": "pt", "sim_eta": "eta_pt"}, axis=1)
        # fake particles are -1 here, was 0 in trackml
        # in root files, background hits don't have a pid, so assign one by grouping on pt
        signal = hits[hits["particle_id"] > 0]
        background = hits[hits["particle_id"] == -1]
        assert len(signal) + len(background) == len(hits)
        background_by_pt = (
            background[["particle_id", "pt"]].value_counts().reset_index()
        )
        background_by_pt["particle_id"] = list(
            range(-1, -len(background_by_pt) - 1, -1)
        )
        background_w_pid = background.drop("particle_id", axis=1).merge(
            background_by_pt, on="pt"
        )
        hits_w_bkg_pid = pd.concat(
            [signal, background_w_pid.drop("count", axis=1)], axis=0
        )
        hits = self.append_cell_features(hits_w_bkg_pid, cells)
        if self.pixel_only:
            hits = hits[hits["volume_id"].isin([1, 2, 3])]
        return self.append_features(hits)

    def load_new_cms_mc_file(self, evt_num):
        # each file contains 1000 events
        file_number = evt_num // 1000 + 1
        ntuple_file = uproot.open(self.indir + "ntuple_" + str(file_number) + ".root")
        # each file will have unique keys
        ntuple_key = list(ntuple_file.keys()[0])
        sub_keys = ntuple_file[ntuple_key].keys()
        hit_key = next(x for x in sub_keys if x.startswith("hits"))
        cell_key = next(x for x in sub_keys if x.startswith("cell"))
        return ntuple_file, ntuple_key, hit_key, cell_key

    def read_cms_run3(self, f):
        err_msg = f"You passed a file {f}, but CMS Run 3 data is not implemented yet"
        raise NotImplementedError(err_msg)
