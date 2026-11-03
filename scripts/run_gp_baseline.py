#!/usr/bin/env python3
"""Train the episode-held-out offline GP baseline on a validated NPZ or UCL tar bundle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from thesis_vla.gp.pipeline import run_gp_experiment

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="gp_chunk_starts.npz or the UCL gp_dataset_bundle.tar.gz")
    parser.add_argument("--output", required=True, help="New/empty directory for result exports")
    parser.add_argument("--config", default=str(ROOT / "configs" / "gp_baseline.json"))
    parser.add_argument("--final-test", action="store_true", help="Evaluate the locked test episodes; omit while inspecting validation/model choices.")
    parser.add_argument("--synthetic-data", action="store_true", help="Label the input as synthetic test data.")
    parser.add_argument("--verified-ucl-pi05", action="store_true", help="Require the documented UCL bundle checksum and 161-pair/20-episode schema before labeling results as real VLA-derived data.")
    args = parser.parse_args()
    result = run_gp_experiment(args.data, args.output, args.config, evaluate_test=args.final_test, synthetic_data=args.synthetic_data, verified_ucl_dataset=args.verified_ucl_pi05)
    print(json.dumps({"output_dir": result["output_dir"], "dataset_label": result["metrics"]["dataset_label"],
        "validation_gp_overall_mae": result["metrics"]["validation"]["gp"]["overall_mae"],
        "validation_ridge_overall_mae": result["metrics"]["validation"]["ridge"]["overall_mae"],
        "final_test_evaluated": args.final_test}, indent=2))
if __name__ == "__main__":
    main()
