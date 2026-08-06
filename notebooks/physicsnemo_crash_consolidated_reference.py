#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyvista as pv
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from physicsnemo.models.geotransolver import GeoTransolver


# configuration

@dataclass
class Config:
    raw_train_dir: Path
    raw_validation_dir: Path
    raw_test_dir: Path

    processed_train_dir: Path
    processed_validation_dir: Path
    processed_test_dir: Path

    stats_path: Path
    checkpoint_path: Path
    predicted_output_dir: Path
    exact_output_dir: Path

    num_time_steps: int = 26
    static_feature_name: str = "thickness"

    slice_num: int = 128
    n_layers: int = 5

    start_lr: float = 1.0e-4
    end_lr: float = 3.0e-7
    epochs: int = 100
    validation_frequency: int = 10

    num_workers: int = 0
    curator_processes: int = 4
    use_amp: bool = True

    @property
    def rollout_steps(self) -> int:
        return self.num_time_steps - 1

    @property
    def functional_dim(self) -> int:
        # initial coordinates (3) + thickness (1)
        return 4

    @property
    def output_dim(self) -> int:
        # future position for every rollout step
        return self.rollout_steps * 3

# preprocessing d3plot files

def run_curator(raw_dir: Path, output_dir: Path, num_processes: int) -> None:
    """Convert LS-DYNA runs to one VTP file per run."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    executable = shutil.which("physicsnemo-curator-etl")
    if executable is None:
        raise RuntimeError(
            "physicsnemo-curator-etl was not found. "
            "Install PhysicsNeMo-Curator before preprocessing."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        executable,
        "--config-dir=examples/structural_mechanics/crash/config",
        "--config-name=crash_etl",
        "serialization_format=vtp",
        f"etl.source.input_dir={raw_dir.resolve()}",
        f"serialization_format.sink.output_dir={output_dir.resolve()}",
        f"etl.processing.num_processes={num_processes}",
    ]

    subprocess.run(command, check=True)


def preprocess_all_splits(cfg: Config) -> None:
    """Preprocess training, validation, and test simulations."""
    run_curator(cfg.raw_train_dir, cfg.processed_train_dir, cfg.curator_processes)
    run_curator(
        cfg.raw_validation_dir,
        cfg.processed_validation_dir,
        cfg.curator_processes,
    )
    run_curator(cfg.raw_test_dir, cfg.processed_test_dir, cfg.curator_processes)

# reading VTP simulations

def natural_sort_key(name: str) -> list[object]:
    """Sort names containing numbers in timestep order."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.findall(r"\d+|\D+", name)
    ]


def find_displacement_fields(mesh: pv.PolyData) -> list[str]:
    """Find and order displacement arrays stored in the VTP."""
    names = [
        name
        for name in mesh.point_data.keys()
        if name.startswith("displacement_t")
    ]

    if not names:
        raise KeyError(
            "No displacement fields were found. Expected names such as "
            "'displacement_t0.000'."
        )

    return sorted(names, key=natural_sort_key)


def read_vtp_trajectory(
    vtp_path: Path,
    num_time_steps: int,
    thickness_name: str,
) -> tuple[np.ndarray, np.ndarray, pv.PolyData]:
    """
    Read one curated simulation.

    Returns:
        positions: [T, N, 3]
        thickness: [N, 1]
        base_mesh: original mesh used for output
    """
    mesh = pv.read(vtp_path)

    reference_positions = np.asarray(mesh.points, dtype=np.float32)
    displacement_names = find_displacement_fields(mesh)

    displacements = [
        np.asarray(mesh.point_data[name], dtype=np.float32)
        for name in displacement_names
    ]

    # Curator VTP contains reference coordinates and displacement fields.
    positions = np.stack(
        [reference_positions + displacement for displacement in displacements],
        axis=0,
    )

    # Keep the configured number of time steps.
    if positions.shape[0] < num_time_steps:
        pad_count = num_time_steps - positions.shape[0]
        padding = np.repeat(positions[-1:], pad_count, axis=0)
        positions = np.concatenate([positions, padding], axis=0)
    else:
        positions = positions[:num_time_steps]

    if thickness_name not in mesh.point_data:
        raise KeyError(
            f"Static feature '{thickness_name}' was not found in {vtp_path.name}."
        )

    thickness = np.asarray(
        mesh.point_data[thickness_name],
        dtype=np.float32,
    )

    if thickness.ndim == 1:
        thickness = thickness[:, None]

    if thickness.shape[0] != reference_positions.shape[0]:
        raise ValueError(
            f"Thickness node count {thickness.shape[0]} does not match "
            f"mesh node count {reference_positions.shape[0]}."
        )

    return positions, thickness, mesh

# normalization statistics

@dataclass
class NormalizationStats:
    position_mean: torch.Tensor
    position_std: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor

    def to(self, device: torch.device) -> "NormalizationStats":
        return NormalizationStats(
            position_mean=self.position_mean.to(device),
            position_std=self.position_std.to(device),
            feature_mean=self.feature_mean.to(device),
            feature_std=self.feature_std.to(device),
        )


def compute_stats(
    trajectories: list[torch.Tensor],
    features: list[torch.Tensor],
) -> NormalizationStats:
    """Compute per-channel statistics from the training split."""
    all_positions = torch.cat(
        [trajectory.reshape(-1, 3) for trajectory in trajectories],
        dim=0,
    )
    all_features = torch.cat(features, dim=0)

    position_mean = all_positions.mean(dim=0)
    position_std = all_positions.std(dim=0, unbiased=False).clamp_min(1.0e-8)

    feature_mean = all_features.mean(dim=0)
    feature_std = all_features.std(dim=0, unbiased=False).clamp_min(1.0e-8)

    return NormalizationStats(
        position_mean=position_mean,
        position_std=position_std,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def save_stats(stats: NormalizationStats, path: Path) -> None:
    """Save training statistics for validation and inference."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "position_mean": stats.position_mean.tolist(),
        "position_std": stats.position_std.tolist(),
        "feature_mean": stats.feature_mean.tolist(),
        "feature_std": stats.feature_std.tolist(),
    }

    path.write_text(json.dumps(payload, indent=2))


def load_stats(path: Path) -> NormalizationStats:
    """Load saved training statistics."""
    payload = json.loads(path.read_text())

    return NormalizationStats(
        position_mean=torch.tensor(payload["position_mean"], dtype=torch.float32),
        position_std=torch.tensor(payload["position_std"], dtype=torch.float32),
        feature_mean=torch.tensor(payload["feature_mean"], dtype=torch.float32),
        feature_std=torch.tensor(payload["feature_std"], dtype=torch.float32),
    )

# simulation samples

@dataclass
class SimulationSample:
    run_name: str
    initial_coordinates: torch.Tensor
    node_features: torch.Tensor
    target_positions: torch.Tensor
    source_path: Path

    def to(self, device: torch.device) -> "SimulationSample":
        return SimulationSample(
            run_name=self.run_name,
            initial_coordinates=self.initial_coordinates.to(device),
            node_features=self.node_features.to(device),
            target_positions=self.target_positions.to(device),
            source_path=self.source_path,
        )


class CrashPointCloudDataset(Dataset[SimulationSample]):
    """
    Position-only point-cloud dataset used by the one-shot GeoTransolver.

    Each simulation produces one sample:
        initial_coordinates: [N, 3]
        node_features:       [N, 1]
        target_positions:    [N, T-1, 3]
    """

    def __init__(
        self,
        data_dir: Path,
        num_time_steps: int,
        thickness_name: str,
        stats: NormalizationStats | None = None,
        fit_stats: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.num_time_steps = num_time_steps
        self.thickness_name = thickness_name

        self.vtp_paths = sorted(data_dir.glob("*.vtp"))

        if not self.vtp_paths:
            raise FileNotFoundError(f"No VTP files found in {data_dir}")

        raw_trajectories: list[torch.Tensor] = []
        raw_features: list[torch.Tensor] = []

        for path in self.vtp_paths:
            positions, thickness, _ = read_vtp_trajectory(
                path,
                num_time_steps,
                thickness_name,
            )
            raw_trajectories.append(torch.from_numpy(positions))
            raw_features.append(torch.from_numpy(thickness))

        if fit_stats:
            self.stats = compute_stats(raw_trajectories, raw_features)
        elif stats is not None:
            self.stats = stats
        else:
            raise ValueError("Provide stats or set fit_stats=True.")

        self.samples: list[SimulationSample] = []

        for path, positions, thickness in zip(
            self.vtp_paths,
            raw_trajectories,
            raw_features,
        ):
            normalized_positions = (
                positions - self.stats.position_mean.view(1, 1, 3)
            ) / self.stats.position_std.view(1, 1, 3)

            normalized_thickness = (
                thickness - self.stats.feature_mean.view(1, -1)
            ) / self.stats.feature_std.view(1, -1)

            sample = SimulationSample(
                run_name=path.stem,
                initial_coordinates=normalized_positions[0],
                node_features=normalized_thickness,
                target_positions=normalized_positions[1:].transpose(0, 1),
                source_path=path,
            )
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SimulationSample:
        return self.samples[index]


def single_sample_collate(batch: list[SimulationSample]) -> SimulationSample:
    """
    Keep batch size at one because simulations have different node counts.
    """
    if len(batch) != 1:
        raise ValueError("This reference pipeline supports batch_size=1 only.")
    return batch[0]

# one-shot GeoTransolver

class CrashGeoTransolverOneShot(nn.Module):
    """
    GeoTransolver wrapper matching the repository's one-shot crash rollout.

    The base network predicts flattened position offsets for every future
    timestep. The offsets are reshaped and added to the initial coordinates.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()

        self.rollout_steps = cfg.rollout_steps

        self.model = GeoTransolver(
            functional_dim=cfg.functional_dim,
            out_dim=cfg.output_dim,
            geometry_dim=3,
            global_dim=None,
            slice_num=cfg.slice_num,
            n_layers=cfg.n_layers,
            use_te=False,
            time_input=False,
            include_local_features=True,
        )

    def forward(self, sample: SimulationSample) -> torch.Tensor:
        coordinates = sample.initial_coordinates
        features = sample.node_features

        local_embedding = torch.cat([coordinates, features], dim=-1)

        flat_prediction = self.model(
            local_embedding=local_embedding.unsqueeze(0),
            geometry=coordinates.unsqueeze(0),
            local_positions=coordinates.unsqueeze(0),
            global_embedding=None,
        ).squeeze(0)

        node_count = coordinates.shape[0]
        needed_width = self.rollout_steps * 3

        if flat_prediction.shape[-1] < needed_width:
            raise ValueError(
                f"Model produced {flat_prediction.shape[-1]} channels, "
                f"but {needed_width} are required."
            )

        position_offsets = flat_prediction[:, :needed_width].reshape(
            node_count,
            self.rollout_steps,
            3,
        )

        predicted_positions = (
            position_offsets + coordinates.unsqueeze(1)
        )

        return predicted_positions

# training and validation

def make_loader(dataset: Dataset[SimulationSample], shuffle: bool) -> DataLoader:
    """Create a variable-node-count dataloader."""
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        collate_fn=single_sample_collate,
    )


@torch.no_grad()
def validate(
    model: CrashGeoTransolverOneShot,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    """Compute overall and per-timestep validation MSE."""
    model.eval()

    total_mse = 0.0
    timestep_mse = None
    sample_count = 0

    for sample in loader:
        sample = sample.to(device)

        prediction = model(sample)
        squared_error = (prediction - sample.target_positions).square()

        total_mse += squared_error.mean().item()

        current_timestep_mse = squared_error.mean(dim=(0, 2)).detach().cpu()
        timestep_mse = (
            current_timestep_mse
            if timestep_mse is None
            else timestep_mse + current_timestep_mse
        )

        sample_count += 1

    model.train()

    if sample_count == 0 or timestep_mse is None:
        raise RuntimeError("Validation dataset is empty.")

    return total_mse / sample_count, timestep_mse / sample_count


def save_checkpoint(
    model: CrashGeoTransolverOneShot,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    path: Path,
) -> None:
    """Save model and training state."""
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        path,
    )


def load_model_checkpoint(
    model: CrashGeoTransolverOneShot,
    path: Path,
    device: torch.device,
) -> None:
    """Load model weights for inference."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])


def train_model(
    cfg: Config,
    train_dataset: CrashPointCloudDataset,
    validation_dataset: CrashPointCloudDataset,
    device: torch.device,
) -> CrashGeoTransolverOneShot:
    """Train the one-shot crash surrogate."""
    train_loader = make_loader(train_dataset, shuffle=True)
    validation_loader = make_loader(validation_dataset, shuffle=False)

    model = CrashGeoTransolverOneShot(cfg).to(device)
    model.train()

    criterion = nn.MSELoss()

    # NVIDIA's full experiment selects Muon; I used Adam here b/c it's more compatible with common PyTorch versions
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.start_lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.end_lr,
    )

    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    for epoch in range(cfg.epochs):
        epoch_loss = 0.0

        for sample in train_loader:
            sample = sample.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(sample)
                loss = criterion(prediction, sample.target_positions)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.detach().item()

        scheduler.step()

        average_loss = epoch_loss / max(len(train_loader), 1)

        print(
            f"epoch={epoch + 1:04d} "
            f"train_mse={average_loss:.6e} "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        should_validate = (
            (epoch + 1) % cfg.validation_frequency == 0
            or epoch + 1 == cfg.epochs
        )

        if should_validate:
            validation_mse, timestep_mse = validate(
                model,
                validation_loader,
                device,
            )

            print(f"validation_mse={validation_mse:.6e}")
            print(
                "validation_mse_by_timestep="
                + np.array2string(
                    timestep_mse.numpy(),
                    precision=4,
                    separator=", ",
                )
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                path=cfg.checkpoint_path,
            )

    return model

# inference and VTP output

def denormalize_positions(
    positions: torch.Tensor,
    stats: NormalizationStats,
) -> torch.Tensor:
    """Return node positions to their original coordinate scale."""
    return (
        positions * stats.position_std.view(1, 1, 3)
        + stats.position_mean.view(1, 1, 3)
    )


def save_vtp_trajectory(
    source_vtp: Path,
    positions: torch.Tensor,
    output_dir: Path,
    field_name: str,
    suffix: str,
) -> None:
    """
    Write one VTP mesh per predicted timestep.

    positions: [N, T, 3]
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    base_mesh = pv.read(source_vtp)
    positions_by_time = positions.transpose(0, 1).detach().cpu().numpy()

    for timestep, timestep_positions in enumerate(positions_by_time):
        if timestep_positions.shape[0] != base_mesh.n_points:
            raise ValueError(
                f"Prediction has {timestep_positions.shape[0]} nodes, "
                f"but source mesh has {base_mesh.n_points}."
            )

        output_mesh = base_mesh.copy(deep=True)
        output_mesh.points = timestep_positions
        output_mesh.point_data[field_name] = timestep_positions

        output_path = output_dir / f"frame_{timestep:03d}_{suffix}.vtp"
        output_mesh.save(output_path)


@torch.no_grad()
def run_inference(
    cfg: Config,
    model: CrashGeoTransolverOneShot,
    test_dataset: CrashPointCloudDataset,
    device: torch.device,
) -> None:
    """Predict every test simulation and export VTP trajectories."""
    model.eval()
    loader = make_loader(test_dataset, shuffle=False)
    stats_on_device = test_dataset.stats.to(device)

    for sample in loader:
        sample = sample.to(device)

        normalized_prediction = model(sample)
        normalized_exact = sample.target_positions

        prediction = denormalize_positions(
            normalized_prediction,
            stats_on_device,
        )
        exact = denormalize_positions(
            normalized_exact,
            stats_on_device,
        )

        predicted_run_dir = cfg.predicted_output_dir / sample.run_name
        exact_run_dir = cfg.exact_output_dir / sample.run_name

        save_vtp_trajectory(
            source_vtp=sample.source_path,
            positions=prediction,
            output_dir=predicted_run_dir,
            field_name="prediction",
            suffix="pred",
        )

        save_vtp_trajectory(
            source_vtp=sample.source_path,
            positions=exact,
            output_dir=exact_run_dir,
            field_name="exact",
            suffix="exact",
        )

        relative_l2 = (
            torch.linalg.vector_norm(prediction - exact)
            / torch.linalg.vector_norm(exact).clamp_min(1.0e-8)
        )

        print(
            f"run={sample.run_name} "
            f"relative_l2={relative_l2.item():.6e} "
            f"output={predicted_run_dir}"
        )

# complete workflow

def build_datasets(
    cfg: Config,
) -> tuple[
    CrashPointCloudDataset,
    CrashPointCloudDataset,
    CrashPointCloudDataset,
]:
    """Create train, validation, and test datasets."""
    train_dataset = CrashPointCloudDataset(
        data_dir=cfg.processed_train_dir,
        num_time_steps=cfg.num_time_steps,
        thickness_name=cfg.static_feature_name,
        fit_stats=True,
    )

    save_stats(train_dataset.stats, cfg.stats_path)

    validation_dataset = CrashPointCloudDataset(
        data_dir=cfg.processed_validation_dir,
        num_time_steps=cfg.num_time_steps,
        thickness_name=cfg.static_feature_name,
        stats=train_dataset.stats,
    )

    test_dataset = CrashPointCloudDataset(
        data_dir=cfg.processed_test_dir,
        num_time_steps=cfg.num_time_steps,
        thickness_name=cfg.static_feature_name,
        stats=train_dataset.stats,
    )

    return train_dataset, validation_dataset, test_dataset


def run_pipeline(
    cfg: Config,
    skip_preprocessing: bool,
    skip_training: bool,
) -> None:
    """Run preprocessing, training, and inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    if not skip_preprocessing:
        preprocess_all_splits(cfg)

    train_dataset, validation_dataset, test_dataset = build_datasets(cfg)

    print(
        f"train_runs={len(train_dataset)} "
        f"validation_runs={len(validation_dataset)} "
        f"test_runs={len(test_dataset)}"
    )

    model = CrashGeoTransolverOneShot(cfg).to(device)

    if skip_training:
        load_model_checkpoint(model, cfg.checkpoint_path, device)
    else:
        model = train_model(
            cfg,
            train_dataset,
            validation_dataset,
            device,
        )

    run_inference(
        cfg,
        model,
        test_dataset,
        device,
    )

# command-line interface

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidated PhysicsNeMo crash GeoTransolver workflow."
    )

    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--raw-validation", type=Path, required=True)
    parser.add_argument("--raw-test", type=Path, required=True)

    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("./processed_vtp"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./outputs"),
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--curator-processes", type=int, default=4)

    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Use VTP files already stored under processed-root.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load the saved checkpoint and run inference only.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processed_root = args.processed_root.resolve()
    output_root = args.output_root.resolve()

    cfg = Config(
        raw_train_dir=args.raw_train.resolve(),
        raw_validation_dir=args.raw_validation.resolve(),
        raw_test_dir=args.raw_test.resolve(),
        processed_train_dir=processed_root / "train",
        processed_validation_dir=processed_root / "validation",
        processed_test_dir=processed_root / "test",
        stats_path=output_root / "stats" / "normalization.json",
        checkpoint_path=output_root / "checkpoints" / "crash_geotransolver.pt",
        predicted_output_dir=output_root / "predicted_vtps",
        exact_output_dir=output_root / "exact_vtps",
        epochs=args.epochs,
        curator_processes=args.curator_processes,
    )

    run_pipeline(
        cfg=cfg,
        skip_preprocessing=args.skip_preprocessing,
        skip_training=args.skip_training,
    )


if __name__ == "__main__":
    main()
