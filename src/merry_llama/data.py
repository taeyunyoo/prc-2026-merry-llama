"""Project-relative paths and dataset definitions for the PRC 2026 pipeline.

Importing this module does not load or modify data. No user-specific paths or
environment names are required. All entry points use this shared definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path


class FlightSet(str, Enum):
    TRAIN = "train"
    RANK = "rank"
    FINAL = "final"


def project_root() -> Path:
    """Locate the repository in an editable installation (pip install -e .)."""
    return Path(__file__).resolve().parents[2]


def training_names() -> list[str]:
    bounds = [date(2025, month, 1) for month in range(1, 13)] + [date(2026, 1, 1)]
    return [f"training_{start.isoformat()}_{end.isoformat()}.parquet"
            for start, end in zip(bounds[:-1], bounds[1:])]


def dataset_names(flight_set: FlightSet | str | None = None) -> list[str]:
    """Return movement datasets; submission templates are not movement data."""
    if flight_set is None or flight_set == "all":
        return training_names() + ["ranking.parquet", "final_ranking.parquet"]
    selected = FlightSet(flight_set)
    if selected is FlightSet.TRAIN:
        return training_names()
    if selected is FlightSet.RANK:
        return ["ranking.parquet"]
    return ["final_ranking.parquet"]


def metadata_donor_names() -> list[str]:
    """Frozen donor scope inherited from the existing v55 preparation."""
    return training_names() + ["ranking.parquet"]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @classmethod
    def from_root(cls, root: Path | str | None = None) -> "ProjectPaths":
        return cls(project_root() if root is None else Path(root))

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def external_dir(self) -> Path:
        return self.data_dir / "external"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def metadata_filled_dir(self) -> Path:
        return self.interim_dir / "metadata_filled"

    @property
    def base_features_dir(self) -> Path:
        return self.interim_dir / "base_features"

    @property
    def model_data_dir(self) -> Path:
        """Shared departure dataset before model-specific preparation."""
        return self.data_dir / "model_dataset"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def submission_dir(self) -> Path:
        return self.data_dir / "submission"

    @property
    def model_dir(self) -> Path:
        return self.root / "models"

    @property
    def config_dir(self) -> Path:
        return self.root / "configs"

    @property
    def report_dir(self) -> Path:
        return self.root / "reports"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    def raw_files(self, flight_set: FlightSet | str | None = None) -> list[Path]:
        return [self.raw_dir / name for name in dataset_names(flight_set)]

