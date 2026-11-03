# %% [markdown]
# # Why the collection ADC cut is 3100 (AbsRS) and 1300 (Simple Threshold)
#
# tp3d gives no 3D point to collection TPs with `adc_integral` below `ADC_CUT_X` (v5 keeps them
# only inside 3D clusters of at least 10 supported points). The cut is there to remove noise and
# radiological TPs. This check uses MC truth to label the collection TPs and shows, for every
# possible cut, which fraction of each category it removes, by number of TPs and by ADC.
#
# Each TP gets the category that deposited most of its energy (`<dataset>_TrackIDEs`):
# - **noise**: no TrackIDE at all;
# - **radiological**: its particle is not in `MCParticles` (the radiological generators are only
#   stored as counts);
# - otherwise the `GeneratorLabel` of the particle: atmospherics, cosmics or marley = **signal**.
#
# The cut values are the "knee" of these curves: almost all noise and most radiological TPs are
# below it, while the signal TPs below it carry less than 1 % of the signal ADC. Truth was used to
# choose them; tp3d itself never uses truth.
#
# Run from this folder (`python adc_cut.py` or cell by cell); it writes `adc_cut.png` and
# `adc_cut_scan.csv` here.

# %%
# Imports

import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path.cwd()
sys.path.insert(0, str(HERE.parent))
import tp3d  # noqa: E402

# %%
# Settings

DATA_DIR = HERE.parent.parent / "data" / "AtmCosRad" / "21373"
FIRST_FILE, LAST_FILE = 1, 2          # file numbers (the _000001.h5 suffix), inclusive: 100 events
DATASETS = ["tpmakerTPCabsRS", "tpmakerTPCsimpleThr"]
NAMES = {"tpmakerTPCabsRS": "AbsRS", "tpmakerTPCsimpleThr": "Simple Threshold"}
SIGNAL = ["atmospherics", "cosmics", "marley"]
CATEGORIES = SIGNAL + ["radiological", "noise"]
COLORS = {"atmospherics": "tab:blue", "cosmics": "tab:orange", "marley": "tab:green",
          "signal": "black", "radiological": "tab:red", "noise": "tab:gray"}
SCAN = [500, 1000, 1300, 1500, 2000, 2500, 3000, 3100, 3500, 4000, 5000, 6000]

# %%
# Load the collection TPs with their category


def files_in_range(data_dir, first, last):
    out = []
    for p in sorted(data_dir.glob("*.h5")):
        m = re.search(r"_(\d+)\.h5$", p.name)
        if m and first <= int(m.group(1)) <= last:
            out.append(p)
    return out


def load_event(event, dataset):
    """Collection TPs of one event: adc_integral and category."""
    mc = event["MCParticles"]
    generator = dict(zip(mc["trackID"][:].ravel(), (g.decode() for g in mc["GeneratorLabel"][:].ravel())))
    tp = pd.DataFrame({k: event[dataset][k][:].ravel() for k in ["ID", "channel", "adc_integral"]})
    tp = tp[tp.channel % tp3d.CH_PER_APA >= 1600]
    ide = pd.DataFrame({k: event[f"{dataset}_TrackIDEs"][k][:].ravel() for k in ["ID", "trackID", "energy"]})
    ide["category"] = ide.trackID.abs().map(generator).fillna("radiological")
    dominant = (ide.groupby(["ID", "category"]).energy.sum().reset_index()
                .sort_values("energy").drop_duplicates("ID", keep="last").set_index("ID").category)
    return pd.DataFrame({"adc": tp.adc_integral.astype(float).values,
                         "category": tp.ID.map(dominant).fillna("noise").values})


files = files_in_range(DATA_DIR, FIRST_FILE, LAST_FILE)
tps, n_events = {}, 0
for ds in DATASETS:
    rows = []
    for path in files:
        with h5py.File(path) as f:
            rows += [load_event(f["Events"][ev], ds) for ev in f["Events"]]
    n_events = len(rows)
    tps[ds] = pd.concat(rows, ignore_index=True)
print(f"{len(files)} files, {n_events} events")
for ds, t in tps.items():
    print(f"{NAMES[ds]}: {len(t)} collection TPs,", t.category.value_counts().to_dict())

# %%
# Scan table: % of each category below the cut (TPs and ADC), and the background left above it


def scan(t, cuts):
    sig = t.category.isin(SIGNAL)
    out = []
    for cut in cuts:
        below = t.adc < cut
        row = {"cut": cut}
        for name, m in [("noise", t.category == "noise"), ("radiological", t.category == "radiological"),
                        ("signal", sig)]:
            row[f"{name} TPs removed %"] = 100 * below[m].mean()
            row[f"{name} ADC removed %"] = 100 * t.adc[m & below].sum() / t.adc[m].sum()
        above = ~below
        row["background TPs left %"] = 100 * (above & ~sig).sum() / above.sum()
        out.append(row)
    return pd.DataFrame(out)


tables = []
for ds in DATASETS:
    s = scan(tps[ds], SCAN).assign(dataset=NAMES[ds])
    s["chosen"] = np.where(s.cut == tp3d.ADC_CUT_X[ds], "<--", "")
    tables.append(s)
    print(f"\n{NAMES[ds]} (tp3d cut: {tp3d.ADC_CUT_X[ds]})")
    print(s.drop(columns="dataset").round(1).to_string(index=False))
pd.concat(tables).round(2).to_csv(HERE / "adc_cut_scan.csv", index=False)

# %% [markdown]
# ## The plot
#
# Top: collection `adc_integral` per category (log-log). Bottom: fraction of each category below
# the cut, by number of TPs (solid) and by ADC (dashed). The vertical line is the tp3d cut.

# %%
bins = np.logspace(1, 6, 101)
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
for col, ds in enumerate(DATASETS):
    t = tps[ds]
    cut = tp3d.ADC_CUT_X[ds]
    ax = axes[0, col]
    for cat in CATEGORIES:
        a = t.adc[t.category == cat]
        ax.hist(a, bins=bins, histtype="step", lw=2, color=COLORS[cat], label=f"{cat} ({len(a)})")
    ax.set_yscale("log")
    ax.set_ylabel(f"collection TPs / {n_events} events")
    ax.set_title(f"{NAMES[ds]}: collection adc_integral by category")
    ax.legend(fontsize=9)
    ax = axes[1, col]
    for cat, m in [("signal", t.category.isin(SIGNAL)), ("radiological", t.category == "radiological"),
                   ("noise", t.category == "noise")]:
        a = np.sort(t.adc[m].values)
        ax.plot(bins, 100 * np.searchsorted(a, bins) / len(a), lw=2, color=COLORS[cat], label=f"{cat}, TPs")
        cum = np.concatenate([[0], np.cumsum(a)]) / a.sum()
        ax.plot(bins, 100 * cum[np.searchsorted(a, bins)], lw=2, ls="--", color=COLORS[cat], label=f"{cat}, ADC")
    ax.set_xscale("log")
    ax.set_xlabel("cut on collection adc_integral")
    ax.set_ylabel("% removed (below the cut)")
    ax.set_ylim(0, 101)
    ax.grid(alpha=0.3)
    s = scan(t, [cut]).iloc[0]
    ax.set_title(f"tp3d cut {cut} removes: noise {s['noise TPs removed %']:.0f} % of TPs, radiological "
                 f"{s['radiological TPs removed %']:.0f} % of TPs,\nsignal {s['signal TPs removed %']:.0f} % "
                 f"of TPs but {s['signal ADC removed %']:.1f} % of its ADC", fontsize=10)
    for ax in axes[:, col]:
        ax.axvline(cut, color="k", lw=1.5, ls=":")
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10)
fig.suptitle(f"Collection ADC cut of tp3d, run 21373 files {FIRST_FILE:06d}-{LAST_FILE:06d} ({n_events} events)")
plt.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(HERE / "adc_cut.png", dpi=120)
plt.show()
