# tp3d v6: 3D points from trigger primitives (dune10kt 1x2x6)

`tp3d` turns the trigger primitives (TPs) of one event into 3D space points, using only the
detector geometry and the TPs themselves (no MC truth at run time). For every collection-plane TP
it finds the U and V TPs that match it in time, height and charge, removes ghost combinations, and
returns one 3D point. The full description and the performance numbers are in
`tp3d_algorithm_v6.txt` (short version first, then the full one); `tp3d_algorithm_v5.txt`
describes everything v6 inherits from v5.

v6 is v5 plus four changes that keep more of the energy (signal = atmospherics + cosmics):

1. a tighter U-V height tolerance for 3-plane matches (3.0 -> 1.25 cm), which removes most ghost
   combinations;
2. a rescue for long collection TPs (tracks heading toward the APA) that the charge test used to
   leave without any match;
3. unsupported points inside groups of at least 5 neighbouring collection TPs become supported;
4. a charge budget: an induction TP shared by several points must have enough charge for all of
   them, which refuses ghosts in showers.

On 1000 new events v6 keeps 84.4 % (AbsRS) / 80.2 % (Simple Threshold) of the atmospheric + cosmic
energy, against 77.8 % / 75.5 % for v5, at slightly higher purity; see Performance.

(The folder keeps its name `tp3d_v5_release`; the code in it is v6.)

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
| `tp3d_algorithm_v6.txt` | how v6 works, how its settings were chosen, and how well it does (truth check) |
| `tp3d_algorithm_v5.txt` | the v5 description (everything v6 did not change) |
| `checks/adc_cut.py`, `.ipynb`, `.png`, `_scan.csv` | why the collection ADC cut is 3100 / 1300: ADC per truth category and the fraction of each category removed for every cut (needs the MC truth tables; not needed to run tp3d) |
| `event_display/` | 3D and projection views of one event, coloured by MC particle (needs truth) |

`tp3d.py` reads the `.json` files from its own folder when it is imported. If one is missing it does
**not** complain: it falls back to an older, worse configuration (and AbsRS without its `calib_`
file stops with "no timing calibration"; the charge budget needs the `qcal_` files). Always copy
the folder as a whole.

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
`50 events, 30443 points (21257 supported)` for AbsRS on run 21373 file 000001.

From Python:

```python
import tp3d

events = tp3d.load_events("FILE.h5", dataset="tpmakerTPCabsRS")   # {event: table of TPs}
for event, tps in events.items():
    points = tp3d.reconstruct(tps, "tpmakerTPCabsRS")
    good = points[points.support]                                   # the points to trust
```

Options of `reconstruct` worth knowing:

- `split_long=2.0`: a collection TP covering more than 2 cm of drift becomes several points along
  its drift length, sharing its charge (column `part`), instead of one point at its peak.
- `budget_q=None`: no charge budget. Faster (AbsRS ~79 instead of ~88 ms/event, Simple Threshold
  ~61 instead of ~100 ms/event, one core) for ~0.3-0.5 points of energy and purity.
- `cut_keep_cluster=None`: plain ADC cut, as in v3. Removes the ~10-20 noise TPs per event kept
  next to real activity, at the price of the small signal hits on tracks.
- `h_tol=3.0, rescue_len=None, promote_cl=None, budget_q=None`: exactly v5.

## Output

One row per 3D point, i.e. per collection TP that found a match and passed the ADC cut (1300 ADC for
Simple Threshold, 3100 for AbsRS; smaller TPs are kept only if they belong to a 3D cluster of at
least 10 supported points):

| column | meaning |
|---|---|
| `x, y, z` | position [cm], detector coordinates |
| `support` | True: accepted through seeds, 3D neighbours, a straight track or a large cluster (trust these); False: isolated best guess |
| `nplanes` | 3 (U, V and X agree) or 2 (only one induction plane) |
| `stage` | how it was accepted: `unique`, `prob_seed` (seeds), `support` (grown from neighbours), `bridge` (along a straight track), `cluster` (v6: in a 2D cluster of >= 5 collection TPs), `best_chi2` (unsupported) |
| `chi2` | matching score: sum of squared pulls of height, time and charge (lower is better) |
| `rescue` | v6: True if the point's candidates were rebuilt without the charge test (long collection TPs) |
| `adc_integral`, `channel`, `time_peak` | of the collection TP |
| `iX`, `i0`, `i1` | row of the X, U and V TP in that event's TP table (-1 = plane missing); `run_tp3d.py` also writes the collection TP's `ID` as `tp_ID` |
| `lenx` | drift length covered by the collection TP [cm] |

For most uses, keep only `support == True`.

## Performance (from `tp3d_algorithm_v6.txt`)

1000 AtmCosRad events, run 21373 files 000268-000287, never used for tuning. Truth is used only
for this check. Signal = atmospherics + cosmics (marley, supernova, is shown on its own).

Each collection TP belongs to one particle, the one with the most electrons in it (`numElectrons`
of the `<dataset>_TrackIDEs`), and has exactly one outcome:

- **kept**: it has a supported point, and its U and V TPs both contain that particle (a missing
  induction plane is accepted);
- **wrong match**: it has a supported point, but the U or V TP does not contain that particle;
- **unsupported**: it has only an unsupported point;
- **no match**: no point, and its ADC is above the cut (no U/V candidate survived);
- **ADC cut**: no point, and its ADC is below the cut.

| AbsRS / Simple Threshold | v5 | v6 |
|---|---|---|
| **signal energy kept** | 77.8 / 75.5 % | **84.4 / 80.2 %** |
| signal energy in wrong matches | 12.7 / 16.9 % | 12.0 / 16.3 % |
| purity: supported ADC that is kept signal | 85.8 / 80.2 % | 86.1 / 80.6 % |
| signal energy of supported points within 5 cm of the true height | 60.7 / 65.0 % | 65.6 / 68.2 % |
| radiological ADC supported | 5.8 / 10.4 % | 8.7 / 13.6 % |
| noise TPs supported per event (next to real activity) | 9.4 / 19.9 | 10.6 / 20.5 |
| iBDM energy kept (10 events) | 84.7 / 78.8 % | 86.8 / 82.1 % |
| time per event (one core, ~10k TPs) | ~79 / ~62 ms | ~88 / ~100 ms |

v6, share of each generator's true deposited energy, AbsRS / Simple Threshold, in %. Primaries are
the generator's own particles, secondaries those made by Geant4.

| | kept | wrong match | unsupported | no match | ADC cut |
|---|---|---|---|---|---|
| **signal (atmospherics + cosmics)** | **84.4 / 80.2** | 12.0 / 16.3 | 1.7 / 1.6 | 0.6 / 0.8 | 1.2 / 1.0 |
| atmospherics | 80.5 / 75.4 | 15.1 / 20.5 | 2.1 / 1.9 | 0.8 / 1.1 | 1.4 / 1.2 |
| – primary | 89.0 / 84.1 | 8.4 / 13.1 | 0.6 / 0.7 | 1.2 / 1.6 | 0.7 / 0.6 |
| – secondary | 73.0 / 67.6 | 21.1 / 27.1 | 3.3 / 3.0 | 0.4 / 0.6 | 2.1 / 1.8 |
| cosmics | 97.0 / 96.4 | 1.9 / 2.6 | 0.5 / 0.5 | 0.1 / 0.1 | 0.6 / 0.5 |
| marley (not signal) | 83.0 / 83.6 | 1.5 / 1.2 | 8.5 / 7.8 | 4.3 / 5.2 | 2.7 / 2.2 |
| radiological | 4.9 / 9.5 | 0.5 / 0.2 | 12.5 / 10.6 | 0.4 / 2.1 | 81.6 / 77.5 |

- Weighted by ADC instead of true energy, signal kept is 86.4 / 82.1 %.
- The largest loss left is wrong matches inside atmospheric showers (secondaries). The position
  score shows they are mostly really misplaced, not only mislabelled; see section B10 of
  `tp3d_algorithm_v6.txt`.
- Two more ideas against shower ghosts were implemented and tested but are off, as they gave no
  gain: track-shaped sharing of induction TPs and cluster-level matching (section B6).
