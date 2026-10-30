"""Revalidate committed evidence and export GP data; no model/GPU required."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thesis_vla.dataset import export_dataset

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("experiment",type=Path)
    args=p.parse_args()
    print(export_dataset(args.experiment))
