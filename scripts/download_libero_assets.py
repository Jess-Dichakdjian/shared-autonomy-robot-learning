"""Fetch the official LIBERO simulation asset dataset into the user cache."""
from pathlib import Path
from huggingface_hub import snapshot_download

destination = Path.home() / ".cache" / "libero" / "assets"
destination.mkdir(parents=True, exist_ok=True)
print(f"Downloading official LIBERO simulation assets to {destination}")
snapshot_download(
    repo_id="lerobot/libero-assets",
    repo_type="dataset",
    local_dir=str(destination),
)
required = ("articulated_objects", "stable_scanned_objects", "turbosquid_objects", "stable_hope_objects")
missing = [name for name in required if not (destination / name).is_dir()]
if missing:
    raise RuntimeError(f"LIBERO asset snapshot is incomplete; missing {missing}")
print("LIBERO_ASSETS_READY")
