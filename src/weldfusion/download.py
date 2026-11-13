"""Download the public datasets from Zenodo, verify MD5 checksums and unpack them into data/raw."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from .paths import Paths
from .sources import CHIANESE, KNAAK, RemoteFile, Source


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(source: Source, f: RemoteFile, dest_dir: Path, retries: int = 5) -> Path:
    """Download one file (resuming partial downloads) and verify its MD5."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f.name
    if dest.exists() and dest.stat().st_size == f.size and md5sum(dest) == f.md5:
        return dest
    for _ in range(retries):
        have = dest.stat().st_size if dest.exists() else 0
        if have > f.size:
            dest.unlink()
            have = 0
        req = urllib.request.Request(source.url(f))
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                mode = "ab" if have and resp.status == 206 else "wb"
                with open(dest, mode) as out:
                    shutil.copyfileobj(resp, out, 1 << 20)
        except OSError as exc:  # network hiccup: retry and resume
            print(f"  retrying {f.name}: {exc}")
            continue
        if dest.stat().st_size == f.size:
            break
    digest = md5sum(dest)
    if digest != f.md5:
        raise RuntimeError(f"MD5 mismatch for {f.name}: {digest} != {f.md5}")
    return dest


def extract_rar(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for cmd in (
        ["unrar", "x", "-o+", "-inul", str(archive), str(dest) + "/"],
        ["7z", "x", "-y", f"-o{dest}", str(archive)],
        ["bsdtar", "-xf", str(archive), "-C", str(dest)],
    ):
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            return
    raise RuntimeError("Extracting RAR archives needs one of: unrar, 7z, bsdtar")


def run(paths: Paths) -> None:
    ch_dir = paths.raw / CHIANESE.key
    for f in CHIANESE.files:
        print(f"[download] {CHIANESE.doi} {f.name}")
        local = fetch(CHIANESE, f, paths.downloads / CHIANESE.key)
        if f.name.endswith(".rar"):
            extract_rar(local, ch_dir)
        else:
            ch_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, ch_dir / f.name)

    kn_dir = paths.raw / KNAAK.key
    for f in KNAAK.files:
        print(f"[download] {KNAAK.doi} {f.name}")
        local = fetch(KNAAK, f, paths.downloads / KNAAK.key)
        with zipfile.ZipFile(local) as zf:
            zf.extractall(kn_dir)
