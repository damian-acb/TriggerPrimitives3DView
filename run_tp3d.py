"""Minimal example: 3D points from the TPs of one HDF5 file with tp3d v5.

usage: python run_tp3d.py FILE.h5 [DATASET]
       DATASET = tpmakerTPCabsRS (default) or tpmakerTPCsimpleThr
       -> FILE_<DATASET>_points.csv, one row per 3D point
"""
import sys
from pathlib import Path

import pandas as pd

import tp3d

path = Path(sys.argv[1])
dataset = sys.argv[2] if len(sys.argv) > 2 else "tpmakerTPCabsRS"

rows = []
for event, tps in tp3d.load_events(path, dataset=dataset).items():
    points = tp3d.reconstruct(tps, dataset)
    points["event"] = event
    points["tp_ID"] = tps.ID.reindex(points.iX).values if "ID" in tps else -1
    rows.append(points)
out = pd.concat(rows, ignore_index=True)
cols = ["event", "tp_ID", "x", "y", "z", "adc_integral", "channel", "time_peak", "nplanes", "support", "stage", "chi2"]
out = out[cols]
dest = path.with_name(f"{path.stem}_{dataset}_points.csv")
out.to_csv(dest, index=False)
print(f"{out.event.nunique()} events, {len(out)} points ({out.support.sum()} supported) -> {dest}")
