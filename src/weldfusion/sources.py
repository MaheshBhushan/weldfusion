"""Public datasets used by WeldFusion (Zenodo records, both CC-BY-4.0)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteFile:
    name: str
    md5: str
    size: int


@dataclass(frozen=True)
class Source:
    key: str
    record: int
    doi: str
    license: str
    citation: str
    modalities: tuple[str, ...]
    files: tuple[RemoteFile, ...]

    def url(self, f: RemoteFile) -> str:
        from urllib.parse import quote

        return f"https://zenodo.org/api/records/{self.record}/files/{quote(f.name)}/content"


CHIANESE = Source(
    key="chianese2022",
    record=6732794,
    doi="10.5281/zenodo.6732794",
    license="CC-BY-4.0",
    citation=(
        "Chianese, G., Sun, T., Franciosa, P., Ceglarek, D., Patalano, S. (2022). "
        "Using photodiodes and supervised machine learning for automatic classification "
        "of weld defects in laser welding of thin foils copper-to-steel battery tabs "
        "[Data set]. Zenodo."
    ),
    modalities=("process_parameters", "photodiode", "quality_label", "penetration_depth"),
    files=(
        RemoteFile("Read me.txt", "ae9b2b3b2a91b70c5d43aec2869c725a", 466),
        RemoteFile("Statistic featuress.xlsx", "8e48790cc8c204a1740de2c3173b7cb1", 30927),
        RemoteFile("Signals - Dataset A.rar", "542201d1996bbde1330a809c2e2e22e3", 13443760),
        RemoteFile("Signals - Dataset B.rar", "725fba467973957da7c296deb8694a7b", 3420430),
        RemoteFile("Signals - Dataset C.rar", "3a1e40d73b0cb6b3acba6cb15b1271eb", 20617766),
    ),
)

KNAAK = Source(
    key="knaak2017",
    record=1049434,
    doi="10.5281/zenodo.1049434",
    license="CC-BY-4.0",
    citation="Knaak, C. (2017). Laser welding IR data for ML [Data set]. Zenodo.",
    modalities=("process_parameters", "coaxial_camera"),
    files=(
        RemoteFile(
            "hdf5_IR-data_clean_labels.zip", "5c0bb8b43cc0470e8da69f3f17ba3a93", 263966519
        ),
    ),
)

SOURCES = (CHIANESE, KNAAK)
