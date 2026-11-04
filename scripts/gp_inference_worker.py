"""Line-oriented worker that loads GPInference in the isolated CPU environment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    try:
        from thesis_vla.gp import GPInference
        predictor = GPInference.load(args.model)
        digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
        print(json.dumps({"ready": True, "model_sha256": digest}), flush=True)
    except Exception as exc:
        print(json.dumps({"ready": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 2

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("op") == "close":
                break
            started = time.perf_counter()
            mean, std = predictor.predict(request.get("state"))
            elapsed = time.perf_counter() - started
            print(json.dumps({"ok": True, "mean": mean.tolist(), "std": std.tolist(),
                              "prediction_seconds": elapsed}), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
