"""3D space points from trigger primitives (dune10kt 1x2x6), without MC truth.

Inputs per TP: channel, time_peak (DTS ticks), adc_integral.  Only detector
geometry and plane timing are used, never backtracked truth.

Geometry (dunecore dune10kt_v6_refactored_1x2x6.gdml, the geometry of
dunefd_1x2x6_services, plus GeoObjectSorterAPA / WireReadoutSorterAPA):
  - 12 APAs in 6 z columns (232.39 cm apart, first column at z = -0.876) and 2
    rows: TPCs are sorted in increasing z, y, x and APA c = TPCs 2c, 2c+1, so even
    APAs are the bottom row (y in [-600.02, 0]) and odd APAs the top row
    (y in [0, 600.02]).
  - wire 0 of every plane is the corner wire on the readout end, which is the
    outer edge of the row (bottom edge for bottom APAs, top edge for top APAs).
  - U and V wires at +-35.71 deg from vertical with 0.4667 cm pitch (0.57477 cm
    along z); collection wires at 0.479 cm pitch, z = 0.5982 .. 230.0392 per APA.
  - collection planes at |x| = 3.002 cm.

Model
-----
Each APA has U (ch 0-799), V (800-1599) and X/collection (1600-2079 on the x<0
face, 2080-2559 on the x>0 face).  Induction wires wrap around the two vertical
z-edges of the frame, so if the frame is unrolled into a cylinder of
circumference 2W (s = z on the x<0 face, s = 2W - z on the x>0 face) every
induction channel is a straight helix, and 2W*cos(theta)/pitch = 800 channels:

    c_U * DZ = s + h*tan(theta) + aU      (mod 2W)
    c_V * DZ = -s + h*tan(theta) + aV     (mod 2W)

h is the height above the readout end of the APA.  Since H*tan(theta) ~ 431 cm
< 2W ~ 460 cm, one induction channel plus the collection wire z gives h without
ambiguity, and h_U == h_V is the 3-plane consistency test.  Odd (top) APAs are
the even ones rotated by 180 deg about x: z -> ZSPAN - z and the readout end is
at the top.  The per-face offsets aU, aV (A_U, C_UV) are sub-cm channel-map
details measured from TP coincidences; they do not depend on the TPG algorithm.

Timing (depends on the TPG algorithm, see CALIB and calibrate.py): induction TPs
peak at a different time than collection TPs of the same charge; the offset,
the matching window and the time resolution are measured without truth from
the induction-collection time-difference distribution of geometrically
consistent (U, V, X) triplets, against a time-shifted accidental sideband.
For AbsRS that distribution has a main peak and a lobe ~8 samples earlier; each
gets its own window (calibrate_windows.py -> win_<dataset>.json) instead of one
wide window that would also cover the empty gap between them.

Ghost removal (candidates + resolve): when several (U, V) pairs fit one collection
TP, the wrong ones (ghosts) are suppressed by
  - charge consistency: r = ln(adc_ind / adc_X) of a real match follows a narrow
    distribution that depends on adc_X (calibrate_charge.py, from collection TPs
    with a single triplet, no truth); matches beyond Q_CUT widths are dropped,
    which also turns more collection TPs into unambiguous seeds;
  - an induction-sharing limit: one induction TP may serve at most MAX_SHARE
    accepted points, so one induction hit cannot hold up a chain of ghosts;
  - 3D neighbourhood support: points are grown out of the unambiguous seeds.
Q_CUT and MAX_SHARE were tuned on the time-shifted sideband of run 21373.

v2 additions (improved_v2, see tp3d_algorithm_v2.txt):
  - cluster consensus: TPs are clustered per plane (neighbouring wires, overlapping
    in time); triplets whose (X, U, V) cluster combination is a minority for their X
    cluster are dropped (CL_FRAC);
  - probability seeds: a collection TP whose best candidate is 3-plane and holds
    >= SEED_PROB of the model probability exp(-chi2/2) of its candidates is a seed;
  - growth until nothing changes (instead of 4 passes);
  - directional bridging: when growth stops, candidates up to BRIDGE_R cm away that
    lie on the local straight track of the accepted points are accepted, and growth
    resumes.
All v2 settings were chosen truth-free (time-shifted sideband, chi2 model).

v3 addition (improved_v3, see tp3d_algorithm_v3.txt): an ADC cut on collection TPs
(ADC_CUT_X) against noise and radiologicals.  By default (CUT_STAGE = "after") the
reconstruction runs on all TPs and the points of collection TPs below the cut are
removed from the output; cutting before matching loses signal.
Unlike the other settings, its values come from the truth-based ADC study in
check_AtmCosRad/main.py.  Calibrations (timing, windows, charge model) are the v2 ones.

v5 additions (improved_v5, see tp3d_algorithm_v5.txt), against gaps along tracks:
  - chain seeds (chain_seeds): best guesses lining up on a straight chain become seeds,
    for tracks parallel to the APA where no TP has a unique match;
  - MAX_SHARE 3 -> 10, for tracks parallel to one set of induction wires;
  - segment growth (SEG_GROWTH): growth distance between the drift-time segments of the
    collection TPs, for tracks along the drift whose long TPs give sparse points;
  - cluster-aware ADC cut (CUT_KEEP_CLUSTER): the cut spares points in large 3D clusters,
    so small hits on tracks are no longer removed;
  - growth reach 6 -> 10 cm (GROW_R).
These were tuned with MC truth (allowed for this version) on separate files.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

CH_PER_APA = 2560
# --- geometry, from the GDML ---------------------------------------------------
APA_DZ = 232.39          # z distance between APA columns [cm]
ZSPAN = 0.5982 + 230.0392  # first + last collection wire z: z-mirror of the top APAs [cm]
THETA = np.radians(35.71)  # induction wire angle from vertical
DZ = 0.57477             # induction wire pitch along z [cm] (0.4667 / cos(theta))
W = 400 * DZ             # APA width seen by the induction wrap [cm]
TAN = np.tan(THETA)
H = 600.019              # active height of one APA [cm]
Y_EDGE = 600.019         # |y| of the readout ends; the two rows meet at y = 0
# per-face channel-map offsets (face 0: x<0, face 1: x>0), from TP coincidences
A_U = {0: -1.3, 1: -0.3}
C_UV = {0: W - 1.07, 1: W + 1.3}          # aU - aV
A_V = {f: A_U[f] - C_UV[f] for f in (0, 1)}
# drift: |x| = VD * time_peak + X0   (LArSoft ConvertTicksToX; X0 ~ collection plane x)
VD, X0 = 0.0025088, 3.00155

# --- timing, per TPG algorithm (TP dataset name) [DTS ticks] ---------------------
# dt:    induction TP peak time minus collection TP peak time, per view
# t_tol: half-width of the time window for an induction match (number, or per view)
# sig_t: time resolution used in the chi2 (number, or per view)
CALIB = {
    "tpmakerTPCsimpleThr": {"dt": {0: -384, 1: -224}, "t_tol": 96, "sig_t": 48},
    "tpmakerTPCabsRS": None,
}
# calibrate.py writes calib_<dataset>.json; a file overrides the default above
for _f in Path(__file__).resolve().parent.glob("calib_*.json"):
    _c = json.loads(_f.read_text())
    _per_view = lambda x: {int(k): v for k, v in x.items()} if isinstance(x, dict) else x
    CALIB[_f.stem[len("calib_"):]] = {k: _per_view(_c[k]) for k in ("dt", "t_tol", "sig_t")}
# --- charge consistency, per TPG algorithm ------------------------------------------
# q: model of r = ln(adc_induction / adc_collection) for real (U, V, X) triplets, as a
#    function of ln(adc_collection): {"lnqX": [bin centres], "mu": {view: [median per
#    bin]}, "sig": {view: [robust width per bin]}}, interpolated (and held flat beyond
#    the outer bins).  calibrate_charge.py writes qcal_<dataset>.json.
for _f in Path(__file__).resolve().parent.glob("qcal_*.json"):
    _c = json.loads(_f.read_text())
    _ds = _f.stem[len("qcal_"):]
    if CALIB.get(_ds) is not None:
        CALIB[_ds] = {**CALIB[_ds], "q": {"lnqX": np.array(_c["lnqX"]),
                                          **{k: {int(v): np.array(x) for v, x in _c[k].items()} for k in ("mu", "sig")}}}
# --- time windows, per TPG algorithm ------------------------------------------------
# windows: {view: [{"c": window centre, "w": half-width, "m": peak mean, "sig": rms,
#    "pen": chi2 penalty}, ...]}
#    [ticks, relative to dt], one entry per peak of the induction-collection time
#    difference; replaces the single t_tol window.  calibrate_windows.py writes
#    win_<dataset>.json.
for _f in Path(__file__).resolve().parent.glob("win_*.json"):
    _c = json.loads(_f.read_text())
    _ds = _f.stem[len("win_"):]
    if CALIB.get(_ds) is not None:
        CALIB[_ds] = {**CALIB[_ds], "windows": {int(v): w for v, w in _c["windows"].items()}}
# --- v3: ADC cut on collection TPs, per TPG algorithm -----------------------------------
# adc_cut_X: collection TPs with adc_integral below this are not used (no 3D point is built
#    around them); induction TPs are all kept, they are only used for matching.  The values
#    are the "knee" of the ADC study in check_AtmCosRad/main.py (run 21373): ~99 % of noise
#    and ~93 % of radiological collection TPs are below them, against ~29 % of the signal
#    collection TPs carrying < 1 % of the signal collection ADC.
ADC_CUT_X = {"tpmakerTPCsimpleThr": 1300, "tpmakerTPCabsRS": 3100}
# CUT_STAGE: "before" matching (faster) or "after" the reconstruction (see reconstruct)
CUT_STAGE = "after"
for _ds, _cut in ADC_CUT_X.items():
    if CALIB.get(_ds) is not None:
        CALIB[_ds] = {**CALIB[_ds], "adc_cut_X": _cut}
CALIB["TriggerPrimitives"] = CALIB["tpmakerTPCsimpleThr"]   # old single-product dumps

# --- ghost removal ----------------------------------------------------------------------
# Q_CUT:     drop an induction match if |r - mu(adc_X)| > Q_CUT * sig(adc_X) (None: no charge cut)
# MAX_SHARE: at most this many accepted 3D points may use the same induction TP
#            (None: no limit)
Q_CUT = 2.0
MAX_SHARE = 10      # v5: was 3 (a long induction TP on a track parallel to its wires
                    # legitimately serves many collection TPs)

def _event_key(name):
    """Sort key for event group names: "7" (single job) or "000001_7" (merged jobs)."""
    return tuple(int(x) for x in name.split("_"))


TP_KEY = ["channel", "time_start", "samples_over_threshold", "samples_to_peak", "adc_integral", "adc_peak"]


def load_events(path, events=None, dataset="TriggerPrimitives"):
    """Unique TPs per event (old dumps repeat a TP once per backtracked IDE).

    Only reconstruction-level columns are read; the truth columns are ignored.
    z is the wire z from the channel map, used for collection channels only.
    dataset is the TP table: "TriggerPrimitives" for single-product dumps, or the producer
    label (e.g. "tpmakerTPCsimpleThr", "tpmakerTPCabsRS") when several were dumped.
    Keys of the returned dict are the event group names: int for plain numbers,
    str for merged files ("000001_7").
    """
    out = {}
    with h5py.File(path) as f:
        for ev in sorted(f["Events"], key=_event_key) if events is None else map(str, events):
            g = f["Events"][ev][dataset]
            # new dumps: one row per TP with an ID (truth in "<dataset>_TrackIDEs");
            # old dumps: one row per TP/TrackIDE pair, collapsed here
            cols = TP_KEY + ["z"] + (["ID"] if "ID" in g else [])
            d = pd.DataFrame({k: g[k][()].ravel() for k in cols}).drop_duplicates(TP_KEY)
            d["time_start"] = d.time_start.astype(np.int64)
            d["time_peak"] = d.time_start + 32 * d.samples_to_peak
            out[int(ev) if ev.isdigit() else ev] = d.reset_index(drop=True)
    return out


def get_calib(calib):
    """calib may be a dataset name or a dict like the CALIB entries."""
    c = CALIB[calib] if isinstance(calib, str) else calib
    if c is None:
        raise ValueError(f"no timing calibration for {calib!r}; run calibrate.py")
    return c


def prepare(tp, calib):
    """Unique TPs with derived fields. tp needs channel, time_peak, adc_integral, z."""
    c = get_calib(calib)
    d = tp.copy()
    d["apa"] = d.channel // CH_PER_APA
    cia = d.channel % CH_PER_APA
    d["view"] = np.where(cia < 800, 0, np.where(cia < 1600, 1, 2))
    d["cw"] = np.where(d.view == 0, cia, np.where(d.view == 1, cia - 800, cia - 1600))
    d["tc"] = d.time_peak - d.view.map({0: c["dt"][0], 1: c["dt"][1], 2: 0})  # time at collection plane
    return d


def apply_adc_cut(d, calib, adc_cut="calib"):
    """Drop collection TPs with adc_integral < adc_cut (v3).  adc_cut="calib" takes the
    calibration's "adc_cut_X" (none if it has none), None keeps everything, a number
    overrides.  Induction TPs are never cut."""
    if isinstance(adc_cut, str):
        adc_cut = get_calib(calib).get("adc_cut_X")
    if adc_cut is None:
        return d
    return d[(d.view != 2) | (d.adc_integral >= adc_cut)]


def _h(c, s, face, view):
    """Height above the APA readout end implied by induction wire c at unrolled position s."""
    a = np.where(face == 0, A_U[0], A_U[1]) if view == 0 else np.where(face == 0, A_V[0], A_V[1])
    q = c * DZ + (-s if view == 0 else s) - a
    return np.mod(q, 2 * W) / TAN


def candidates(d, calib, h_tol=3.0, time_shift=0, t_tol=None, q_cut=Q_CUT):
    """All geometrically allowed (X, U, V) combinations for every collection TP.

    Returns one row per candidate with the X-TP index, the U/V TP indices (-1 if
    that plane is missing), h and a chi2-like score.  time_shift moves the
    induction planes in time: with a large shift every match is accidental,
    which measures the fake rate without truth.  t_tol overrides the calibrated
    window(s) with one symmetric window (used by calibrate.py to look at a wide window).
    If the calibration has "windows" (calibrate_windows.py), an induction TP is
    matched when it falls in any of them: the main peak and the secondary lobe of
    the induction-collection time difference, instead of one wide window.

    If the calibration has a charge model ("q"), r = ln(adc_ind / adc_X) is
    computed for every induction match and compared with its expected value for
    that collection ADC: matches with |r - mu| > q_cut * sig are dropped
    (q_cut=None keeps them) and ((r - mu) / sig)^2 is added to chi2.
    """
    cal = get_calib(calib)
    windows = cal.get("windows") if t_tol is None else None
    t_tol = cal["t_tol"] if t_tol is None else t_tol
    t_tol = t_tol if isinstance(t_tol, dict) else {0: t_tol, 1: t_tol}
    sig_t = cal["sig_t"] if isinstance(cal["sig_t"], dict) else {0: cal["sig_t"], 1: cal["sig_t"]}
    if windows is None:
        windows = {v: [{"c": 0, "w": t_tol[v], "sig": sig_t[v], "pen": 0.0}] for v in (0, 1)}
    q = cal.get("q")
    cols = ["iX", "i0", "i1", "apa", "nplanes", "h", "x", "y", "z", "chi2", "dt0", "dt1", "r0", "r1", "lenx"]
    out = []
    for apa, g in d.groupby("apa"):
        X = g[g.view == 2]
        if len(X) == 0:
            continue
        face = (X.cw.values >= 480).astype(int)
        zl = X.z.values - APA_DZ * (apa // 2)
        zp = zl if apa % 2 == 0 else ZSPAN - zl
        s = np.where(face == 0, zp, 2 * W - zp)
        tx = X.tc.values
        per_view = {}
        for v in (0, 1):
            I = g[g.view == v].sort_values("tc")
            ti = I.tc.values + time_shift
            # one or more (non-overlapping) time windows around dt; the time part of
            # the chi2 is measured from the centre of the window the match falls in
            ixs, jjs, c2s = [], [], []
            for wd in windows[v]:
                lo = np.searchsorted(ti, tx + wd["c"] - wd["w"])
                hi = np.searchsorted(ti, tx + wd["c"] + wd["w"], side="right")
                n = hi - lo
                ix = np.repeat(np.arange(len(X)), n)
                jj = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if n.sum() else np.zeros(0, int)
                ixs.append(ix); jjs.append(jj)
                c2s.append(((ti[jj] - tx[ix] - wd.get("m", wd["c"])) / wd["sig"]) ** 2 + wd["pen"])
            ix, jj, c2 = np.concatenate(ixs), np.concatenate(jjs), np.concatenate(c2s)
            h = _h(I.cw.values[jj], s[ix], face[ix], v)
            r = np.log(I.adc_integral.values[jj] / X.adc_integral.values[ix])
            # chi2 contribution of this plane: time (above), and charge if calibrated
            ok = h <= H + h_tol
            if q is not None:
                lq = np.log(X.adc_integral.values[ix])
                pull = (r - np.interp(lq, q["lnqX"], q["mu"][v])) / np.interp(lq, q["lnqX"], q["sig"][v])
                c2 = c2 + pull ** 2
                if q_cut is not None:
                    ok &= np.abs(pull) <= q_cut
            per_view[v] = pd.DataFrame({"ix": ix[ok], f"i{v}": I.index.values[jj][ok], f"h{v}": h[ok],
                                        f"dt{v}": (ti[jj] - tx[ix])[ok], f"r{v}": r[ok], f"c2{v}": c2[ok]})
        u, vv = per_view[0], per_view[1]
        trip = u.merge(vv, on="ix")
        trip = trip[np.abs(trip.h0 - trip.h1) < h_tol]
        trip = trip.assign(h=(trip.h0 + trip.h1) / 2, nplanes=3,
                           chi2=((trip.h0 - trip.h1) / 1.4) ** 2 + trip.c20 + trip.c21)
        has3 = set(trip.ix)
        du = u[~u.ix.isin(has3)].assign(i1=-1, dt1=np.nan, r1=np.nan, h=lambda q: q.h0, nplanes=2,
                                        chi2=lambda q: q.c20 + 9)
        dv = vv[~vv.ix.isin(has3)].assign(i0=-1, dt0=np.nan, r0=np.nan, h=lambda q: q.h1, nplanes=2,
                                          chi2=lambda q: q.c21 + 9)
        c = pd.concat([trip, du, dv], ignore_index=True)
        if len(c) == 0:
            continue
        c["h"] = c.h.clip(0, H)
        c["iX"] = X.index.values[c.ix]
        c["apa"] = apa
        c["x"] = np.where(face[c.ix] == 0, -1, 1) * (VD * tx[c.ix] + X0)
        c["y"] = -Y_EDGE + c.h if apa % 2 == 0 else Y_EDGE - c.h
        c["z"] = X.z.values[c.ix]
        c["lenx"] = VD * 32 * X.samples_over_threshold.values[c.ix]     # drift length of the X TP [cm]
        out.append(c[cols])
    if not out:
        return pd.DataFrame(columns=cols)
    c = pd.concat(out, ignore_index=True)
    c["i0"] = c.i0.fillna(-1).astype(int)
    c["i1"] = c.i1.fillna(-1).astype(int)
    return c


def _accept(ok, used, max_share):
    """One candidate per collection TP from `ok` (sorted best first), such that no
    induction TP ends up in more than max_share accepted points.  `used` counts
    the accepted points per induction TP index and is updated in place.
    """
    if max_share is None:
        return ok.drop_duplicates("iX")
    acc = []
    while len(ok):
        free = np.ones(len(ok), bool)
        for col in ("i0", "i1"):
            i = ok[col].values
            free &= (i < 0) | (used[np.maximum(i, 0)] < max_share)
        ok = ok[free]
        best = ok.drop_duplicates("iX")
        if len(best) == 0:
            break
        # several collection TPs may compete for the last free slots of an induction TP:
        # the better-ranked ones get them, the others try their next candidate
        take = np.ones(len(best), bool)
        for col in ("i0", "i1"):
            i = best[col].values
            rank = best.groupby(col).cumcount().values
            take &= (i < 0) | (used[np.maximum(i, 0)] + rank < max_share)
        best = best[take]
        for col in ("i0", "i1"):
            i = best[col].values
            np.add.at(used, i[i >= 0], 1)
        acc.append(best)
        ok = ok[~ok.iX.isin(best.iX)]
    return pd.concat(acc) if acc else ok.iloc[:0]


# --- v2 ghost-removal settings (tuned on the time-shifted sideband, see tp3d_algorithm_v2.txt) ---
# CL_FRAC:  cluster consensus.  Triplets whose (X, U, V) cluster combination has fewer than
#           CL_FRAC of the votes of the best combination of the same X cluster are dropped
#           (only for X clusters of at least CL_MIN TPs; None: off)
# CL_TOL:   time tolerance [ticks] for two TPs on neighbouring wires to be in one cluster
# SEED_PROB: extra seeds: a collection TP whose best candidate is 3-plane and has a model
#           probability >= SEED_PROB among its candidates is a seed too, with
#           P_best = 1 / sum_j exp(-(chi2_j - chi2_best) / 2)  (the time and charge terms
#           of chi2 use calibrated widths).  With two candidates, 0.9 means a chi2 gap of
#           4.4.  None: only unique seeds.
# BRIDGE_R: directional bridging: an unsupported candidate up to BRIDGE_R cm from an
#           accepted point is accepted if it lies within BRIDGE_PERP cm of the local track
#           line through that point (None: off).  The line is the principal axis of the
#           accepted points within BRIDGE_NB cm, used only where they are linear
#           (largest eigenvalue fraction >= BRIDGE_LIN, at least 4 points).
CL_FRAC, CL_MIN, CL_TOL = 0.35, 5, 64
SEED_PROB = 0.9
BRIDGE_R, BRIDGE_PERP, BRIDGE_NB, BRIDGE_LIN = 30.0, 2.5, 8.0, 0.9

# --- v5 settings (tuned with truth on run 21373 files 000223-000227, see tp3d_algorithm_v5.txt) ---
# SEG_GROWTH: growth measures the distance between two points as the distance between the
#           drift-time segments of their collection TPs (a TP spans lenx cm in x around its
#           peak), not between their peaks, so long TPs on tracks along the drift connect.
# CHAIN_*:  chain seeds: after resolve, unsupported TPs whose best-chi2 point lies on a
#           locally straight chain of other points (>= CHAIN_K points within CHAIN_R cm,
#           largest-eigenvalue fraction >= CHAIN_LIN) keep only that candidate, which makes
#           them seeds, and resolve is run again (CHAIN_ROUNDS times at most).  None: off.
# CUT_KEEP_CLUSTER: the ADC cut spares supported points that belong to a 3D cluster of at
#           least this many supported points (linked within CUT_LINK cm): small hits on
#           tracks and showers stay, isolated radiological/noise blips go.  None: plain cut.
SEG_GROWTH = True
# GROW_R:   growth reach [cm] (v2-v3: 6).  10 cm recovers points just beyond 6 cm (big hits,
#           sparse stretches of tracks and showers) for a small purity cost; see the txt.
GROW_R = 10.0
CHAIN_R, CHAIN_K, CHAIN_LIN, CHAIN_ROUNDS = 8.0, 5, 0.9, 4
CUT_KEEP_CLUSTER, CUT_LINK = 10, 6.0


def clusters(d, tol=CL_TOL):
    """Cluster id per TP: connected TPs on neighbouring wires of the same plane (and X
    face) of one APA whose [time_start, time_end] intervals overlap within tol ticks.
    Induction wires are neighbours modulo 800 (they wrap around the APA frame)."""
    t0 = d.time_start.values
    t1 = t0 + 32 * d.samples_over_threshold.values
    plane = np.where(d.view.values < 2, d.view.values, 2 + (d.cw.values >= 480))
    n_w = np.where(d.view.values < 2, 800, 10 ** 6)
    key = pd.DataFrame({"i": np.arange(len(d)), "g": d.apa.values * 4 + plane, "w": d.cw.values,
                        "n": n_w, "t0": t0, "t1": t1})
    nxt = key.assign(w=np.where(key.n == 800, (key.w - 1) % 800, key.w - 1))   # TP seen from wire+1
    m = key.merge(nxt, on=["g", "w"], suffixes=("", "_b"))
    m = m[(m.t0 <= m.t1_b + tol) & (m.t0_b <= m.t1 + tol)]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = coo_matrix((np.ones(len(m)), (m.i.values, m.i_b.values)), shape=(len(d), len(d)))
    return connected_components(g, directed=False)[1]


def cluster_consensus(c, d, frac=CL_FRAC, min_size=CL_MIN):
    """Drop 3-plane candidates whose (X, U, V) cluster combination is a minority for
    their X cluster.  A real track gives many triplets that all combine the same three
    clusters; a ghost combines a hit of the right X cluster with hits of other U/V
    clusters and collects few votes.  Every collection TP keeps at least its
    best-voted triplet, so this only removes alternatives."""
    if frac is None or len(c) == 0:
        return c
    cl = clusters(d)
    pos = pd.Series(np.arange(len(d)), index=d.index)
    c = c.copy()
    c["cX"] = cl[pos.reindex(c.iX).values]
    t = c.nplanes == 3
    c["cU"] = np.where(t, cl[pos.reindex(c.i0.where(t, c.iX)).values], -1)
    c["cV"] = np.where(t, cl[pos.reindex(c.i1.where(t, c.iX)).values], -1)
    size = pd.Series(cl).value_counts()
    tr = c[t]
    votes = tr.groupby(["cX", "cU", "cV"]).size().rename("votes")
    tr = tr.join(votes, on=["cX", "cU", "cV"])
    tr["score"] = tr.votes / tr.groupby("cX").votes.transform("max")
    best = tr.groupby("iX").score.transform("max")
    big = size.reindex(tr.cX).values >= min_size
    drop = tr.index[big & (tr.score < frac) & (tr.score < best)]
    return c.drop(index=drop).drop(columns=["cX", "cU", "cV"])


def _directions(pts, nb=BRIDGE_NB, k=12):
    """Local principal axis and linearity of the accepted points around each point."""
    if len(pts) < 4:                                          # too few points for a direction
        return np.zeros((len(pts), 3)), np.zeros(len(pts)), np.zeros(len(pts))
    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=min(k, len(pts)), distance_upper_bound=nb)
    ok = np.isfinite(dist)
    idx = np.where(ok, idx, 0)
    nbr = pts[idx]                                            # (n, k, 3)
    w = ok[..., None].astype(float)
    n = w.sum(1)
    mean = (nbr * w).sum(1) / n
    dev = (nbr - mean[:, None, :]) * w
    cov = np.einsum("nki,nkj->nij", dev, dev) / n[..., None]
    ev, evec = np.linalg.eigh(cov)
    lin = ev[:, 2] / np.maximum(ev.sum(1), 1e-9)
    return evec[:, :, 2], lin, n[:, 0]


def _bridge(rest, chosen, used, max_share, r=BRIDGE_R, perp=BRIDGE_PERP, lin_min=BRIDGE_LIN):
    """Candidates of `rest` that continue a locally straight track of `chosen`."""
    pts = chosen[["x", "y", "z"]].values
    direc, lin, n = _directions(pts)
    good = (lin >= lin_min) & (n >= 4)
    if not good.any() or len(rest) == 0:
        return rest.iloc[:0]
    tree = cKDTree(pts[good])
    q = rest[["x", "y", "z"]].values
    dist, idx = tree.query(q, k=min(8, good.sum()), distance_upper_bound=r)
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]
    okk = np.isfinite(dist)
    idx = np.where(okk, idx, 0)
    a, dvec = pts[good][idx], direc[good][idx]               # (m, k, 3)
    v = q[:, None, :] - a
    along = np.einsum("mki,mki->mk", v, dvec)
    pd_ = np.linalg.norm(v - along[..., None] * dvec, axis=2)
    pd_ = np.where(okk, pd_, np.inf)
    rest = rest.assign(dperp=pd_.min(1))
    ok = rest[rest.dperp <= perp].sort_values(["dperp", "chi2"])
    return _accept(ok, used, max_share).drop(columns="dperp")


def _growth_dist(rest, chosen, radius, seg):
    """Distance from each remaining candidate to the nearest accepted point (inf beyond
    radius).  seg: measure along x between the drift-time segments of the two TPs."""
    tree = cKDTree(chosen[["x", "y", "z"]].values)
    q = rest[["x", "y", "z"]].values
    if not seg:
        return tree.query(q, distance_upper_bound=radius)[0]
    lc, lr = chosen.lenx.values, rest.lenx.values
    reach = radius + 0.5 * (lc.max() + lr.max())
    k = min(8, len(chosen))
    dist, idx = tree.query(q, k=k, distance_upper_bound=reach)
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]
    ok = np.isfinite(dist)
    idx = np.where(ok, idx, 0)
    a = chosen[["x", "y", "z"]].values[idx]                    # (m, k, 3)
    dx = np.maximum(np.abs(q[:, None, 0] - a[..., 0]) - 0.5 * (lr[:, None] + lc[idx]), 0)
    dyz2 = (q[:, None, 1] - a[..., 1]) ** 2 + (q[:, None, 2] - a[..., 2]) ** 2
    d = np.where(ok, np.sqrt(dx ** 2 + dyz2), np.inf).min(1)
    return np.where(d < radius, d, np.inf)


def resolve(c, radius=GROW_R, n_iter=1000, max_share=MAX_SHARE, seed_prob=SEED_PROB, bridge=BRIDGE_R,
            bridge_perp=BRIDGE_PERP, seg_growth=SEG_GROWTH):
    """Choose one candidate per collection TP using 3D neighbourhood support.

    1. Seeds: collection TPs with exactly one 3-plane candidate ("unique"), plus, if
       seed_prob is set, collection TPs whose best candidate is 3-plane with model
       probability >= seed_prob among their candidates ("prob_seed").
    2. Growth: every other TP takes its candidate closest to an accepted point within
       `radius` cm (GROW_R = 10 in v5, 6 before); repeated until nothing new is accepted
       (at most n_iter passes).
       With seg_growth (v5) the distance is taken between the drift-time segments of
       the two collection TPs, so long TPs on tracks along the drift connect.
    3. Bridging (if bridge is set): TPs still left take a candidate up to `bridge` cm
       from an accepted point if it lies on the local straight track through that
       point ("bridge"); then growth resumes.  2 and 3 alternate until nothing changes.
    No induction TP may be used by more than max_share accepted points.  TPs never
    reached keep their best-chi2 candidate and are flagged support = False.
    """
    c = c.copy()
    c["ncand"] = c.groupby("iX").iX.transform("size")
    used = np.zeros(int(max(c.i0.max(), c.i1.max(), 0)) + 1, int)
    seeds = c[(c.ncand == 1) & (c.nplanes == 3)].sort_values("chi2")
    chosen = [_accept(seeds, used, max_share).assign(stage="unique")]
    if seed_prob is not None:
        s = c[c.ncand > 1].sort_values(["iX", "chi2"])
        rel = np.exp(-(s.chi2 - s.groupby("iX").chi2.transform("min")) / 2)
        s = s.assign(prob=1 / rel.groupby(s.iX).transform("sum"))
        first = s.groupby("iX").head(1)
        g = first[(first.nplanes == 3) & (first.prob >= seed_prob) & ~first.iX.isin(chosen[0].iX)]
        chosen.append(_accept(g.drop(columns="prob").sort_values("chi2"), used, max_share).assign(stage="prob_seed"))
    chosen = pd.concat(chosen)
    rest = c[~c.iX.isin(chosen.iX)]
    it = 0
    while it < n_iter and len(rest) and len(chosen):
        rest = rest.assign(dnn=_growth_dist(rest, chosen, radius, seg_growth))
        ok = _accept(rest[rest.dnn < radius].sort_values(["dnn", "chi2"]), used, max_share)
        rest = rest.drop(columns="dnn")
        stage = "support"
        if len(ok) == 0 and bridge is not None:
            ok = _bridge(rest, chosen, used, max_share, r=bridge, perp=bridge_perp)
            stage = "bridge"
        if len(ok) == 0:
            break
        chosen = pd.concat([chosen, ok.drop(columns="dnn", errors="ignore").assign(stage=stage)])
        rest = rest[~rest.iX.isin(ok.iX)]
        it += 1
    left = rest.sort_values("chi2").drop_duplicates("iX").assign(stage="best_chi2")
    res = pd.concat([chosen, left], ignore_index=True)
    res["support"] = res.stage != "best_chi2"
    return res.drop(columns=["dnn"], errors="ignore")


def chain_seeds(c, p, resolve_kw, r=CHAIN_R, kmin=CHAIN_K, lin=CHAIN_LIN, rounds=CHAIN_ROUNDS):
    """v5: promote unsupported TPs whose best-chi2 point lies on a locally straight chain of
    other points (supported or best-chi2) to seeds, by keeping only that candidate, and
    re-run resolve.  Ghosts scatter; the best guesses along an ambiguous track (e.g. one
    parallel to the APA, where every TP has many equally good candidates) line up."""
    for _ in range(rounds):
        cand = p[p.nplanes == 3]
        pts = cand[["x", "y", "z"]].values
        if len(pts) < kmin:
            break
        dist, nn = cKDTree(pts).query(pts, k=min(16, len(pts)), distance_upper_bound=r)
        ok = np.isfinite(dist)
        nn = np.where(ok, nn, 0)
        w = ok[..., None]
        n = w.sum(1)
        mean = (pts[nn] * w).sum(1) / n
        dev = (pts[nn] - mean[:, None]) * w
        ev = np.linalg.eigvalsh(np.einsum("nki,nkj->nij", dev, dev) / n[..., None])
        straight = (ev[:, 2] / np.maximum(ev.sum(1), 1e-9) >= lin) & (n[:, 0] >= kmin)
        promote = cand[straight & ~cand.support.values]
        if len(promote) == 0:
            break
        c = pd.concat([c[~c.iX.isin(promote.iX)],
                       c.merge(promote[["iX", "i0", "i1"]], on=["iX", "i0", "i1"])], ignore_index=True)
        p = resolve(c, **resolve_kw)
    return p


def cluster_sizes(p, link=CUT_LINK):
    """Size of the 3D cluster (supported points linked within `link` cm) of every point;
    0 for unsupported points."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    out = np.zeros(len(p), int)
    sup = np.flatnonzero(p.support.values)
    if len(sup) == 0:
        return out
    pairs = cKDTree(p[["x", "y", "z"]].values[sup]).query_pairs(r=link, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(sup), len(sup)))
    lab = connected_components(g, directed=False)[1]
    out[sup] = np.bincount(lab)[lab]
    return out


def reconstruct(tp, calib, h_tol=3.0, radius=GROW_R, time_shift=0, q_cut=Q_CUT, max_share=MAX_SHARE,
                cl_frac=CL_FRAC, seed_prob=SEED_PROB, bridge=BRIDGE_R, bridge_perp=BRIDGE_PERP, n_iter=1000,
                adc_cut="calib", cut_stage=CUT_STAGE, seg_growth=SEG_GROWTH, chain=True,
                cut_keep_cluster=CUT_KEEP_CLUSTER):
    """Full chain: unique TPs -> 3D points (one per matched collection TP).

    calib is the TP dataset name ("tpmakerTPCsimpleThr", "tpmakerTPCabsRS",
    "TriggerPrimitives") or a timing dict like the CALIB entries.
    q_cut, max_share, cl_frac, seed_prob and bridge are the ghost-removal settings (see
    the constants above); None switches each one off.  n_iter=4 with
    cl_frac=seed_prob=bridge=None reproduces 3d_test_improved.
    adc_cut: ADC cut on collection TPs (v3, see apply_adc_cut); adc_cut=None reproduces
    improved_v2.  cut_stage "before": the cut TPs are removed before matching (faster);
    "after": the reconstruction runs on all TPs and the points of cut collection TPs are
    removed from the output (the small TPs still help the growth step).
    v5: seg_growth (drift-segment growth distance), chain (chain seeds), max_share=10 and
    cut_keep_cluster (the ADC cut spares points in 3D clusters of at least this size);
    seg_growth=False, chain=False, max_share=3, cut_keep_cluster=None, radius=6 reproduces v3.
    """
    d = prepare(tp, calib)
    if cut_stage == "before":
        d = apply_adc_cut(d, calib, adc_cut)
    c = candidates(d, calib, h_tol=h_tol, time_shift=time_shift, q_cut=q_cut)
    if len(c) == 0:
        return c
    c = cluster_consensus(c, d, frac=cl_frac)
    kw = dict(radius=radius, n_iter=n_iter, max_share=max_share, seed_prob=seed_prob, bridge=bridge,
              bridge_perp=bridge_perp, seg_growth=seg_growth)
    p = resolve(c, **kw)
    if chain:
        p = chain_seeds(c, p, kw)
    p["adc_integral"] = d.adc_integral.reindex(p.iX).values
    p["channel"] = d.channel.reindex(p.iX).values
    p["time_peak"] = d.time_peak.reindex(p.iX).values
    if cut_stage == "after":
        keep = p.iX.isin(apply_adc_cut(d, calib, adc_cut).index).to_numpy(copy=True)
        if cut_keep_cluster is not None:
            keep |= cluster_sizes(p) >= cut_keep_cluster
        p = p[keep].reset_index(drop=True)
    return p
