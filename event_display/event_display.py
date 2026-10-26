"""Event display of tp3d (v5) space points coloured by MC particle: an interactive 3D view and
three 2D projections (z-y, z-x, x-y).

The reconstruction is truth-free; truth is only used to colour the points. Each 3D point gets the
particle (|trackID|) that deposited most of the energy in its collection TP, from the
<dataset>_TrackIDEs table; points of the same particle share a colour in all four views. The
N_SHOW particles with the most ADC in supported points get their own colour and legend entry, the
rest of the signal is grey ("other particles"). TrackIDEs whose |trackID| is not in MCParticles
are radiologicals (shown together in dark red), TPs without TrackIDEs are noise (black). If an
event has no MCParticles at all (e.g. the iBDM file), every trackID is shown by number.

Labels give PDG, generator, initial total energy E_tot (MCParticle InitialEnergy, including the
rest mass, so a nucleus shows ~37 GeV), trackID and whether the particle is a primary (made by
the generator) or a secondary (made by Geant4). The files have no mother information: primaries
are recognised from the trackID numbering, since Geant4 numbers each generator's particles first,
as one block of consecutive trackIDs (see check_energy_retention/energy_retention.py,
mark_primaries). For atmospherics the hover also gives the neutrino interaction (MCNeutrino).

Only supported points are drawn by default; the unsupported ones are in the 3D legend (click to
show) and light grey in the projections.

usage: python event_display.py FILE.h5 EVENT [DATASET ...] [--out DIR]
       DATASET: tpmakerTPCabsRS and/or tpmakerTPCsimpleThr (default: both)
       --out DIR: where to create the output folder (default: this script's folder)
       -> <out>/<file name>_ev<EVENT>/3d_<DATASET>.html        interactive 3D (open in a browser)
                                     views_<DATASET>.png       z-y, z-x and x-y projections, whole detector
                                     views_<DATASET>_zoom.png  the same, zoomed on the event
                                     particles_<DATASET>.csv   supported points and ADC per particle
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TP3D_DIR = HERE.parent / "tp3d_v5_release"           # folder with tp3d.py and its .json calibrations
sys.path.insert(0, str(TP3D_DIR))
import tp3d  # noqa: E402

DATASETS = ["tpmakerTPCabsRS", "tpmakerTPCsimpleThr"]
N_SHOW = 20
# 20 distinguishable colours (d3 category20, darker shade first)
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf",
           "#bcbd22", "#7f7f7f", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
           "#f7b6d2", "#9edae5", "#dbdb8d", "#393b79"]
OTHER, RADIOLOGICAL, NOISE, UNSUPPORTED = "#c7c7c7", "#8b0000", "#000000", "#dddddd"

# active volume of dune10kt_v6_refactored_1x2x6 (volTPCActive*, dunecore GDML) [cm]
DET = {"x": (-363.376, 363.376), "y": (-600.019, 600.019), "z": (-0.876, 1393.464)}
APA_Z = [-0.876 + 232.39 * i for i in range(7)]       # APA column boundaries in z

PDG_NAMES = {11: "e-", -11: "e+", 13: "mu-", -13: "mu+", 15: "tau-", -15: "tau+", 22: "gamma",
             2212: "p", 2112: "n", 211: "pi+", -211: "pi-", 111: "pi0", 321: "K+", -321: "K-",
             130: "K0L", 310: "K0S", 3122: "Lambda", 12: "nu_e", -12: "anti-nu_e", 14: "nu_mu",
             -14: "anti-nu_mu", 16: "nu_tau", -16: "anti-nu_tau"}
ELEMENTS = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
            11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca"}
# simb::MCNeutrino Mode and InteractionType codes
MODES = {0: "QE", 1: "RES", 2: "DIS", 3: "COH", 4: "COH elastic", 5: "nu-e elastic", 6: "IMD annihilation",
         7: "inverse beta decay", 8: "Glashow resonance", 9: "AMNuGamma", 10: "MEC", 11: "diffractive",
         12: "EM", 13: "weak mix"}
INTERACTIONS = {1000: "unspecified", 1001: "CCQE", 1002: "NCQE",
                1003: "CC RES nu p pi+", 1004: "CC RES nu n pi0", 1005: "CC RES nu n pi+",
                1006: "NC RES nu p pi0", 1007: "NC RES nu p pi-", 1008: "NC RES nu n pi0",
                1009: "NC RES nu n pi+", 1010: "CC RES nubar n pi-", 1011: "CC RES nubar p pi0",
                1012: "CC RES nubar p pi-", 1013: "NC RES nubar p pi0", 1014: "NC RES nubar p pi-",
                1015: "NC RES nubar n pi0", 1016: "NC RES nubar n pi+", 1091: "CC DIS", 1092: "NC DIS",
                1096: "NC COH pi", 1097: "CC COH pi", 1098: "nu-e elastic", 1099: "inverse muon decay"}


def pdg_name(pdg):
    if pdg in PDG_NAMES:
        return PDG_NAMES[pdg]
    if abs(pdg) > 1000000000:                          # nucleus 10LZZZAAAI
        z, a = (abs(pdg) // 10000) % 1000, (abs(pdg) // 10) % 1000
        return f"{ELEMENTS.get(z, f'Z{z}')}{a}"
    return str(pdg)


def neutrino_text(nu):
    """One line per MCNeutrino row: flavour, energy, CC/NC, mode and interaction type."""
    if nu is None or len(nu) == 0:
        return ""
    out = []
    for r in nu.itertuples():
        cc = "CC" if r.CCNC == 0 else "NC"
        mode = MODES.get(int(r.Mode), f"mode {int(r.Mode)}")
        it = INTERACTIONS.get(int(r.InteractionType), f"interaction type {int(r.InteractionType)}")
        extra = "" if it in ("unspecified", f"{cc}{mode}", f"{cc} {mode}") else f" ({it})"
        out.append(f"{pdg_name(int(r.NuPDG))} {r.NuEnergy:.3g} GeV {cc} {mode}{extra}")
    return "; ".join(out)


def mark_primaries(mc):
    """True for the generator's own particles: per generator, the run of consecutive trackIDs that
    starts at its lowest trackID (Geant4 numbers all generator particles before any secondary)."""
    m = mc.sort_values(["generator", "trackID"])
    run = ((m.generator != m.generator.shift()) | (m.trackID.diff() != 1)).cumsum()
    return (run == run.groupby(m.generator).transform("min")).reindex(mc.index)


def truth(path, event, dataset):
    """Dominant |trackID| per TP ID, the particle table and the event's neutrino text.

    In the returned dominant trackIDs, -1 = radiological (not in MCParticles), and TPs without
    TrackIDEs are left out (noise)."""
    with h5py.File(path) as f:
        e = f["Events"][str(event)]
        if f"{dataset}_TrackIDEs" not in e:
            raise KeyError(f"event {event} has no {dataset}_TrackIDEs table: this file has no truth")
        g = e["MCParticles"]
        mc = pd.DataFrame({"trackID": g["trackID"][()].ravel(), "PDG": g["PDG"][()].ravel(),
                           "energy": g["InitialEnergy"][()].ravel(),
                           "generator": [x.decode() if isinstance(x, bytes) else x
                                         for x in g["GeneratorLabel"][()].ravel()]})
        ide = pd.DataFrame({k: e[f"{dataset}_TrackIDEs"][k][()].ravel() for k in ("ID", "trackID", "energy")})
        nu = (pd.DataFrame({k: e["MCNeutrino"][k][()].ravel() for k in e["MCNeutrino"]})
              if "MCNeutrino" in e else None)
    mc = mc.drop_duplicates("trackID")
    mc["primary"] = mark_primaries(mc) if len(mc) else []
    nu_text = neutrino_text(nu)
    interaction = {"atmospherics": nu_text, "cosmics": "cosmic ray", "marley": "MARLEY nu-Ar"}
    mc["interaction"] = mc.generator.map(interaction).fillna("")
    mc = mc.set_index("trackID")
    ide["tid"] = ide.trackID.abs()
    if len(mc):                                        # no MCParticles at all: keep every trackID
        ide.loc[~ide.tid.isin(mc.index), "tid"] = -1
    dom = (ide.groupby(["ID", "tid"]).energy.sum().reset_index()
           .sort_values("energy").drop_duplicates("ID", keep="last").set_index("ID").tid)
    return dom, mc, nu_text


def particle_label(t, mc, short=False):
    """Legend / hover text of particle t (t > 0), radiologicals (-1) or noise (-2)."""
    if t == -1:
        return "radiological"
    if t == -2:
        return "noise"
    if t not in mc.index:
        return f"trackID {t}"
    r = mc.loc[t]
    kind = "primary" if r.primary else "secondary"
    if short:
        return f"{pdg_name(int(r.PDG))} E_tot {r.energy:.3g} GeV, {r.generator} {kind} (ID {t})"
    return (f"{pdg_name(int(r.PDG))} ({r.generator}, {kind}), trackID {t}, E_tot = {r.energy:.4g} GeV"
            + (f"<br>{r.interaction}" if r.interaction else ""))


def reconstruct_event(path, event, dataset):
    """tp3d points of one event with the dominant particle of each point ("tid", -1 radiological,
    -2 noise) and its colour ("color"); plus the particle table and the neutrino text."""
    tp = tp3d.load_events(path, events=[event], dataset=dataset)
    tp = next(iter(tp.values()))
    dom, mc, nu_text = truth(path, event, dataset)
    p = tp3d.reconstruct(tp, dataset)
    p["tid"] = tp.ID.reindex(p.iX).map(dom).fillna(-2).astype(int).values
    p["adc_integral"] = p.adc_integral.astype(float)
    sup = p[p.support]
    order = sup[sup.tid > 0].groupby("tid").adc_integral.sum().sort_values(ascending=False).index
    colors = {t: PALETTE[i] for i, t in enumerate(order[:N_SHOW])}
    p["color"] = [colors.get(t, OTHER) if t > 0 else (RADIOLOGICAL if t == -1 else NOISE) for t in p.tid]
    p["shown"] = p.tid.isin(colors.keys())
    return p, mc, nu_text, list(colors)


def particle_table(p, mc):
    """Supported points per particle, most ADC first."""
    s = p[p.support].groupby("tid").agg(points=("x", "size"), adc=("adc_integral", "sum"),
                                         three_plane=("nplanes", lambda n: int((n == 3).sum())))
    s["particle"] = [particle_label(t, mc, short=True) for t in s.index]
    return s.sort_values("adc", ascending=False)


# --- interactive 3D (plotly) ------------------------------------------------------------------

def detector_traces():
    """Outline of the active volume, the APA plane at x = 0 and the APA boundaries."""
    (x0, x1), (y0, y1), (z0, z1) = DET["x"], DET["y"], DET["z"]
    seg = []
    for x in (x0, x1):                                 # the two CPA faces
        seg += [[(x, z0, y0), (x, z1, y0), (x, z1, y1), (x, z0, y1), (x, z0, y0)]]
    for z in (z0, z1):
        for y in (y0, y1):
            seg += [[(x0, z, y), (x1, z, y)]]
    apa = [[(0, z0, y0), (0, z1, y0), (0, z1, y1), (0, z0, y1), (0, z0, y0)], [(0, z0, 0), (0, z1, 0)]]
    apa += [[(0, z, y0), (0, z, y1)] for z in APA_Z[1:-1]]
    out = []
    for name, segs, color in [("detector (active volume)", seg, "#555555"), ("APAs (x = 0)", apa, "#9a9a9a")]:
        xs, ys, zs = [], [], []
        for sgm in segs:
            for a, b, c in sgm:
                xs.append(a); ys.append(b); zs.append(c)
            xs.append(None); ys.append(None); zs.append(None)
        out.append(dict(type="scatter3d", mode="lines", name=name, x=xs, y=ys, z=zs,
                        line=dict(color=color, width=2), hoverinfo="skip"))
    return out


def figure_3d(p, mc, title):
    """Plotly figure as a dict (data + layout). plotly axes: x = drift x, y = z, z = height y."""
    def trace(q, name, color, size, visible=True):
        hover = [f"{particle_label(t, mc)}<br>ch {c}, adc {a:.0f}, {n}-plane"
                 for t, c, a, n in zip(q.tid, q.channel, q.adc_integral, q.nplanes)]
        return dict(type="scatter3d", mode="markers", name=name, visible=visible,
                    x=q.x.round(2).tolist(), y=q.z.round(2).tolist(), z=q.y.round(2).tolist(),
                    marker=dict(size=size, color=color, line=dict(width=0)), text=hover, hoverinfo="text")

    sup = p[p.support]
    data = detector_traces()
    for t in sup[sup.shown].groupby("tid").adc_integral.sum().sort_values(ascending=False).index:
        q = sup[sup.tid == t]
        data.append(trace(q, f"{particle_label(t, mc, short=True)}: {len(q)} pts, {int(q.adc_integral.sum())} ADC",
                          q.color.iloc[0], 3))
    rest = sup[(sup.tid > 0) & ~sup.shown]
    data.append(trace(rest, f"other particles ({rest.tid.nunique()}): {len(rest)} pts", OTHER, 2))
    for t, name in ((-1, "radiological"), (-2, "noise")):
        q = sup[sup.tid == t]
        data.append(trace(q, f"{name}: {len(q)} pts", RADIOLOGICAL if t == -1 else NOISE, 2))
    q = p[~p.support]
    data.append(trace(q, f"unsupported points (all): {len(q)}", "#bbbbbb", 1.5, "legendonly"))
    span = {k: v[1] - v[0] for k, v in DET.items()}
    m = max(span.values())
    layout = dict(title=dict(text=title, font=dict(size=13)), margin=dict(l=0, r=0, t=40, b=0),
                  legend=dict(x=0, y=1, font=dict(size=11)),
                  scene=dict(aspectmode="manual",
                             aspectratio=dict(x=span["x"] / m, y=span["z"] / m, z=span["y"] / m),
                             xaxis=dict(title="x drift [cm]", range=list(DET["x"])),
                             yaxis=dict(title="z [cm]", range=list(DET["z"])),
                             zaxis=dict(title="y [cm]", range=list(DET["y"]))))
    return dict(data=data, layout=layout)


def write_html(fig, path, title):
    path.write_text(
        '<!doctype html><html><head><meta charset="utf-8">'
        f"<title>{title}</title>"
        '<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>'
        "<style>body{margin:0;font-family:sans-serif} #p{width:100vw;height:100vh}</style></head>"
        f'<body><div id="p"></div><script>Plotly.newPlot("p", {json.dumps(fig["data"])}, '
        f'{json.dumps(fig["layout"])});</script></body></html>')


# --- 2D projections (matplotlib) --------------------------------------------------------------

def figure_views(p, mc, title, zoom=False):
    """z-y (looking along the drift), z-x (from above) and x-y (along the beam) projections."""
    views = [("z", "y"), ("z", "x"), ("x", "y")]
    sup = p[p.support]
    lim = {k: v for k, v in DET.items()}
    if zoom and len(sup):
        for k in lim:                                  # signal points, with a 30 cm margin
            v = sup[sup.tid > 0][k] if (sup.tid > 0).any() else sup[k]
            lim[k] = (max(DET[k][0], v.min() - 30), min(DET[k][1], v.max() + 30))
    widths = [lim[a][1] - lim[a][0] for a, _ in views]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.8), gridspec_kw={"width_ratios": widths})
    for ax, (a, b) in zip(axes, views):
        q = p[~p.support]
        ax.scatter(q[a], q[b], s=1, c=UNSUPPORTED, label=f"unsupported ({len(q)})", zorder=1)
        for t, name, size in ((-1, "radiological", 2), (-2, "noise", 2)):
            q = sup[sup.tid == t]
            ax.scatter(q[a], q[b], s=size, c=RADIOLOGICAL if t == -1 else NOISE, label=f"{name} ({len(q)})", zorder=2)
        q = sup[(sup.tid > 0) & ~sup.shown]
        ax.scatter(q[a], q[b], s=3, c=OTHER, label=f"other particles ({len(q)})", zorder=2)
        for t in sup[sup.shown].groupby("tid").adc_integral.sum().sort_values(ascending=False).index:
            q = sup[sup.tid == t]
            ax.scatter(q[a], q[b], s=5, c=q.color.iloc[0], label=f"{particle_label(t, mc, short=True)} ({len(q)})",
                       zorder=3)
        if a == "z":
            for zz in APA_Z:
                ax.axvline(zz, color="#cccccc", lw=0.6, zorder=0)
        if "x" in (a, b):                              # APA plane at x = 0
            (ax.axvline if a == "x" else ax.axhline)(0, color="#999999", lw=0.8, zorder=0)
        if b == "y":
            ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)
        ax.set_xlim(*lim[a])
        ax.set_ylim(*lim[b])
        ax.set_aspect("equal")
        ax.set_xlabel(f"{a} [cm]")
        ax.set_ylabel(f"{b} [cm]")
        ax.set_title(f"{a}-{b}")
    # legend: particles by ADC (as drawn last), then the grey / red / black groups
    handles, labels = axes[0].get_legend_handles_labels()
    order = list(range(4, len(labels))) + [3, 1, 2, 0][:min(4, len(labels))]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=4,
               fontsize=8, markerscale=2.5)
    fig.suptitle(title, fontsize=11)
    n_rows = int(np.ceil(len(labels) / 4))
    fig.tight_layout(rect=(0, min(0.45, 0.035 * n_rows + 0.02), 1, 0.95))
    return fig


# --- everything for one event -----------------------------------------------------------------

def make(path, event, dataset, outdir):
    """Write 3d_<dataset>.html, views_<dataset>.png, views_<dataset>_zoom.png and
    particles_<dataset>.csv into outdir."""
    path = Path(path)
    p, mc, nu_text, _ = reconstruct_event(path, event, dataset)
    title = (f"{path.name}  event {event}  {dataset}" + (f"  |  {nu_text}" if nu_text else "")
             + f"  (tp3d v5, collection adc >= {tp3d.ADC_CUT_X.get(dataset)}; colour = MC particle)")
    outdir.mkdir(parents=True, exist_ok=True)
    write_html(figure_3d(p, mc, title), outdir / f"3d_{dataset}.html", f"{path.stem} ev{event} {dataset}")
    for zoom, suffix in ((False, ""), (True, "_zoom")):
        fig = figure_views(p, mc, title, zoom=zoom)
        fig.savefig(outdir / f"views_{dataset}{suffix}.png", dpi=110)
        plt.close(fig)
    table = particle_table(p, mc)
    table.to_csv(outdir / f"particles_{dataset}.csv")
    return p, table


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("file")
    ap.add_argument("event")
    ap.add_argument("datasets", nargs="*", default=DATASETS)
    ap.add_argument("--out", default=str(HERE), help="where to create the output folder")
    args = ap.parse_args()
    path = Path(args.file)
    outdir = Path(args.out) / f"{path.stem}_ev{args.event}"
    pd.set_option("display.width", 200)
    for ds in args.datasets:
        p, table = make(path, args.event, ds, outdir)
        print(f"\n{ds}: {len(p)} points ({int(p.support.sum())} supported) -> {outdir}")
        print(table.head(10).to_string())


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()
