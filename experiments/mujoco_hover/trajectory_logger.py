"""Small per-episode image and JSONL logger for the Panda MuJoCo scene."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class EpisodeLogger:
    def __init__(self, episode_dir: str | Path, jpeg_quality: int = 90):
        self.episode_dir = Path(episode_dir)
        self.static_dir = self.episode_dir / "static"
        self.wrist_dir = self.episode_dir / "wrist"
        self.static_dir.mkdir(parents=True, exist_ok=False)
        self.wrist_dir.mkdir(parents=True, exist_ok=False)
        self.frames = (self.episode_dir / "frames.jsonl").open("w", encoding="utf-8")
        self.jpeg_quality = int(jpeg_quality)
        self.frame_id = 0

    def log(self, row: dict[str, Any], static_rgb: np.ndarray, wrist_rgb: np.ndarray) -> None:
        """Save one synchronized camera pair and its state/action record."""
        stem = f"{self.frame_id:06d}.jpg"
        options = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        static_path = self.static_dir / stem
        wrist_path = self.wrist_dir / stem
        if not cv2.imwrite(str(static_path), cv2.cvtColor(static_rgb, cv2.COLOR_RGB2BGR), options):
            raise OSError(f"Could not write {static_path}")
        if not cv2.imwrite(str(wrist_path), cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR), options):
            raise OSError(f"Could not write {wrist_path}")

        record = dict(row)
        record["frame_id"] = self.frame_id
        record["static_image"] = static_path.relative_to(self.episode_dir).as_posix()
        record["wrist_image"] = wrist_path.relative_to(self.episode_dir).as_posix()
        self.frames.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.frames.flush()
        self.frame_id += 1

    def close(self) -> None:
        if not self.frames.closed:
            self.frames.close()
