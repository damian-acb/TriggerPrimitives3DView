# tp3d v5: 3D points from trigger primitives (dune10kt 1x2x6)

`tp3d` turns the trigger primitives (TPs) of one event into 3D space points, using only the
detector geometry and the TPs themselves (no MC truth at run time). For every collection-plane TP
it finds the U and V TPs that match it in time, height and charge, removes ghost combinations, and
returns one 3D point. The full description and the performance numbers are in
`tp3d_algorithm_v5.txt`.

v5 is v3 plus five changes that close gaps along tracks (chain seeds, a looser induction sharing
limit, growth that accounts for the drift length of each TP and reaches 10 cm instead of 6, and an
ADC cut that keeps small hits on tracks). It reconstructs ~5-8 % more of the signal charge than v3
at nearly the same purity; see Performance.

## Files

Keep all of these in the same folder:

| file | what it is |
|---|---|
| `tp3d.py` | the algorithm |
| `calib_tpmakerTPCabsRS.json` | AbsRS timing offsets |
| `win_tpmakerTPCabsRS.json` | AbsRS time windows |
| `qcal_tpmakerTPCabsRS.json` | AbsRS charge-consistency model |
| `qcal_tpmakerTPCsimpleThr.json` | Simple Threshold charge-consistency model |
| `run_tp3d.py` | example: one HDF5 file in, one CSV of 3D points out |
| `tp3d_algorithm_v5.txt` | how it works, and how well (truth check) |

`tp3d.py` reads the `.json` files from its own folder when it is imported. If one is missing it does
**not** complain: it falls back to an older, worse configuration (and AbsRS without its `calib_`
file stops with "no timing calibration"). Always copy the folder as a whole.

## Requirements

Python 3 with `numpy`, `pandas`, `scipy` and `h5py` (tested with Python 3.14, numpy 2.5,
pandas 3.0, scipy 1.18, h5py 3.16).

## Input

HDF5 files written by `TriggerPrimitiveHDF5Dump` (dunetrigger), with the TPs of each event in
`Events/<event>/<dataset>`, where `<dataset>` is `tpmakerTPCabsRS` or `tpmakerTPCsimpleThr`, and
the columns `channel`, `time_start`, `samples_over_threshold`, `samples_to_peak`, `adc_integral`,
`adc_peak`, `z` and `ID`. No truth tables are needed.

- Geometry: **dune10kt 1x2x6 only** (dunecore `dune10kt_v6_refactored_1x2x6`).
- The calibration files were measured on run 21373 (AtmCosRad simulation, current TPG settings).
  With a different TPG configuration they should be re-measured.

## Running it

From the command line:

```
python run_tp3d.py FILE.h5                       # AbsRS TPs
python run_tp3d.py FILE.h5 tpmakerTPCsimpleThr   # Simple Threshold TPs
```

This writes `FILE_<dataset>_points.csv` next to the input file and prints a summary, e.g.
`50 events, 30467 points (19749 supported)` for AbsRS (about 4.5 s for a 50-event file).

From Python:

```python
import tp3d

events = tp3d.load_events("FILE.h5", dataset="tpmakerTPCabsRS")   # {event: table of TPs}
for event, tps in events.items():
    points = tp3d.reconstruct(tps, "tpmakerTPCabsRS")
    good = points[points.support]                                   # the points to trust
```

Options of `reconstruct` worth knowing:

- `cut_keep_cluster=None`: plain ADC cut, as in v3. Removes the ~10-20 noise TPs per event that v5
  keeps next to real activity, at the price of the small signal hits on tracks.
- `radius=6`: the growth reach of v3 (10 cm by default), slightly purer, slightly less efficient.
- `seg_growth=False, chain=False, max_share=3, cut_keep_cluster=None, radius=6`: exactly v3 (v5 only
  adds the output column `lenx`, the drift length of the collection TP in cm).

## Output

One row per 3D point, i.e. per collection TP that found a match and passed the ADC cut (1300 ADC for
Simple Threshold, 3100 for AbsRS; smaller TPs are kept only if they belong to a 3D cluster of at
least 10 supported points):

| column | meaning |
|---|---|
| `x, y, z` | position [cm], detector coordinates |
| `support` | True: accepted through seeds, 3D neighbours or a straight track (trust these); False: isolated best guess |
| `nplanes` | 3 (U, V and X agree) or 2 (only one induction plane) |
| `stage` | how it was accepted: `unique`, `prob_seed` (seeds), `support` (grown from neighbours), `bridge` (along a straight track), `best_chi2` (unsupported) |
| `chi2` | matching score: sum of squared pulls of height, time and charge (lower is better) |
| `adc_integral`, `channel`, `time_peak` | of the collection TP |
| `iX`, `i0`, `i1` | row of the X, U and V TP in that event's TP table (-1 = plane missing); `run_tp3d.py` also writes the collection TP's `ID` as `tp_ID` |
| `lenx` | drift length covered by the collection TP [cm] |

For most uses, keep only `support == True`.

## Performance (from `tp3d_algorithm_v5.txt`, 1000 AtmCosRad events, not used for tuning)

| | AbsRS | Simple Threshold |
|---|---|---|
| signal charge in correctly matched supported points | 86.9 % (v3: 78.8 %) | 87.5 % (v3: 82.1 %) |
| cosmics / marley | 90.5 % / 69 % | 93.5 % / 73 % |
| supported charge that is correctly matched signal (purity) | 94.1 % (v3: 94.7 %) | 90.6 % (v3: 91.2 %) |
| supported points that are correctly matched signal (by count) | 77 % | 71 % |
| noise TPs kept per event (next to real activity) | ~10 | ~21 |
| time per event (one CPU core, ~10k TPs) | ~78 ms | ~69 ms |

"Signal" = atmospheric neutrinos, cosmics and marley; the remaining supported points are mostly real
radiological deposits above the ADC cut, a few wrong matches and the noise TPs above.

### What happens to each collection TP, one particle per TP

The same 1000 events (run 21373, files 000001-000020), with truth used only for the check. Each
collection TP belongs to one particle, the one with the most electrons in it (`numElectrons` of the
`<dataset>_TrackIDEs`), and has exactly one outcome:

- **kept**: it has a supported point, and its U and V TPs both contain that particle (a missing
  induction plane is accepted);
- **wrong match**: it has a supported point, but the U or V TP does not contain that particle;
- **unsupported**: it has only an unsupported point;
- **no match**: no point, and its ADC is above the cut (no U/V candidate survived);
- **ADC cut**: no point, and its ADC is below the cut.

Share of each generator's collection-TP ADC, AbsRS / Simple Threshold, in %. Primaries are the
generator's own particles, secondaries those made by Geant4.

| | kept | wrong match | unsupported | no match | ADC cut |
|---|---|---|---|---|---|
| **signal** | **82.0 / 79.7** | 9.1 / 14.4 | 3.5 / 3.1 | 4.8 / 2.3 | 0.5 / 0.5 |
| atmospherics | 79.8 / 76.0 | 11.5 / 18.0 | 3.2 / 2.8 | 4.9 / 2.5 | 0.6 / 0.6 |
| – primary | 84.5 / 84.9 | 5.9 / 9.2 | 1.9 / 1.7 | 7.4 / 3.9 | 0.3 / 0.3 |
| – secondary | 74.6 / 66.2 | 17.8 / 27.9 | 4.6 / 4.0 | 2.1 / 1.0 | 0.9 / 0.9 |
| cosmics | 90.4 / 93.4 | 0.7 / 1.5 | 3.8 / 3.7 | 4.8 / 1.2 | 0.3 / 0.3 |
| – primary | 90.5 / 93.6 | 0.6 / 1.4 | 3.8 / 3.6 | 4.8 / 1.2 | 0.3 / 0.3 |
| – secondary | 79.3 / 72.7 | 10.1 / 18.4 | 8.2 / 7.0 | 1.1 / 0.6 | 1.3 / 1.3 |
| marley | 68.6 / 72.8 | 1.7 / 1.4 | 23.4 / 19.7 | 4.7 / 4.5 | 1.6 / 1.6 |
| radiological | 5.2 / 10.0 | 0.6 / 0.3 | 23.0 / 17.7 | 0.6 / 2.6 | 70.6 / 69.4 |

- Weighted by true deposited energy instead of ADC, signal kept is 80.3 / 78.1 %; by number of
  TPs, 60.5 / 62.7 % (the ADC cut removes 20 / 17 % of the signal TPs, but only 0.5 % of their ADC).
- Purity: 88.8 / 82.6 % of the supported ADC is kept signal (73.5 / 67.8 % by number of points).
- This is stricter than "correctly matched" in the table above (86.9 / 87.5 %), which accepts a
  point when its U and V TPs share *any* particle with the collection TP. The difference is mostly
  inside showers, hence the larger wrong-match share of atmospheric secondaries.
