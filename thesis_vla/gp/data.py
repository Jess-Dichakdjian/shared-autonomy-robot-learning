"""Validated, pickle-free loader for the GP chunk-start dataset contract."""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import tarfile
import hashlib
import json
import numpy as np
from numpy.typing import NDArray
@dataclass(frozen=True)
class GPDataset:
    """Primary normalized state/action pairs and their episode identities."""
    X: NDArray[np.float64]
    Y: NDArray[np.float64]
    episode_id: NDArray[np.int64]
    source: str
def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}
def _read_archive(path: Path) -> dict[str, np.ndarray]:
    with tarfile.open(path, mode="r:gz") as archive:
        member = next((item for item in archive.getmembers() if item.isfile() and item.name.removeprefix("./") == "dataset/gp_chunk_starts.npz"), None)
        if member is None:
            raise ValueError(f"{path} does not contain dataset/gp_chunk_starts.npz")
        stream = archive.extractfile(member)
        if stream is None:
            raise ValueError("Could not read GP NPZ member from archive")
        npz_bytes = stream.read()
        checksum_member = next((item for item in archive.getmembers() if item.isfile() and item.name.removeprefix("./") == "dataset/sha256.json"), None)
        if checksum_member is not None:
            checksum_stream = archive.extractfile(checksum_member)
            manifest = json.load(checksum_stream) if checksum_stream is not None else {}
            expected = manifest.get("gp_chunk_starts.npz")
            if expected is not None and hashlib.sha256(npz_bytes).hexdigest() != expected:
                raise ValueError("GP NPZ checksum does not match dataset/sha256.json")
        with np.load(BytesIO(npz_bytes), allow_pickle=False) as bundle:
            return {key: bundle[key] for key in bundle.files}
def load_gp_dataset(path: str | Path) -> GPDataset:
    """Load a benchmark NPZ or gp_dataset_bundle.tar.gz and validate X/Y/IDs."""
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "dataset" / "gp_chunk_starts.npz"
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.name.endswith(".tar.gz"):
        arrays = _read_archive(source)
    elif source.suffix == ".npz":
        arrays = _read_npz(source)
    else:
        raise ValueError("Dataset input must be an .npz or .tar.gz file")
    missing = {"X", "Y", "episode_id"}.difference(arrays)
    if missing:
        raise ValueError(f"Dataset is missing required arrays: {sorted(missing)}")
    X, Y, episode_id = (arrays[name] for name in ("X", "Y", "episode_id"))
    if X.ndim != 2 or X.shape[1] != 8:
        raise ValueError(f"X must have shape (N, 8); got {X.shape}")
    if Y.ndim != 2 or Y.shape[1] != 7:
        raise ValueError(f"Y must have shape (N, 7); got {Y.shape}")
    if X.shape[0] == 0 or Y.shape[0] != X.shape[0]:
        raise ValueError("X and Y must have the same non-zero sample count")
    if episode_id.ndim != 1 or episode_id.shape[0] != X.shape[0]:
        raise ValueError("episode_id must be one-dimensional with one ID per row")
    if not np.issubdtype(X.dtype, np.number) or not np.issubdtype(Y.dtype, np.number):
        raise ValueError("X and Y must be numeric arrays")
    if not np.issubdtype(episode_id.dtype, np.integer):
        raise ValueError("episode_id must use an integer dtype")
    X, Y = np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)
    episode_id = np.asarray(episode_id, dtype=np.int64)
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("X and Y must contain only finite values")
    return GPDataset(X=X, Y=Y, episode_id=episode_id, source=str(source))
