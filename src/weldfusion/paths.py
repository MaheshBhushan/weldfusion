from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Paths:
    """Directory layout of one pipeline run. `raw` can point anywhere (e.g. a test fixture)."""

    root: Path
    raw: Path

    @classmethod
    def at(cls, root: str | Path, raw: str | Path | None = None) -> Paths:
        root = Path(root)
        return cls(root=root, raw=Path(raw) if raw else root / "data" / "raw")

    @property
    def downloads(self) -> Path:
        return self.root / "data" / "downloads"

    @property
    def fused(self) -> Path:
        return self.root / "data" / "fused"

    @property
    def features(self) -> Path:
        return self.root / "data" / "features"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


def load_params(path: str | Path = "params.yaml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
